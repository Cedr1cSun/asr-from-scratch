"""SenseVoice 风格 SANM encoder + CTC,全随机初始化从零训。

结构参照 FunAudioLLM/SenseVoice model.py(SenseVoiceEncoderSmall /
EncoderLayerSANM / MultiHeadedAttentionSANM);LFR 与 Sinusoidal PE 公式
vendor 自 funasr,自含实现,不引入 funasr 依赖。缩小版结构与忠实度边界
见 docs/superpowers/specs/2026-07-03-sensevoice-from-scratch-design.md。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    ParakeetFeatureExtractor,
    ParakeetTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput

from asrfs.common.ctc import CTCProcessorBundle

TOKENIZER_SOURCE = "nvidia/parakeet-ctc-0.6b"

LOSS_FAMILY = "ctc"
# LABEL_PAD_ID 必须等于 CTC blank id(= tokenizer.vocab_size)。tokenizer 来自
# Hub(TOKENIZER_SOURCE),import 时无网络/无缓存即无法解析真值,故硬编码
# nvidia/parakeet-ctc-0.6b 的已知值 1024。守卫:
# tests/test_sensevoice_adapter.py::test_label_pad_id_matches_tokenizer(slow)。
LABEL_PAD_ID = 1024
EXPECTED_FROZEN: set = set()

SMALL_MODEL = dict(
    hidden_size=384,
    num_hidden_layers=16,
    num_attention_heads=4,
    intermediate_size=1536,
    fsmn_kernel_size=11,
    sanm_shift=0,
    lfr_m=7,
    lfr_n=6,
    num_mel_bins=80,
    dropout=0.0,
)


class SenseVoiceConfig(PretrainedConfig):
    model_type = "sensevoice-ctc"

    def __init__(
        self,
        vocab_size=1025,
        hidden_size=384,
        num_hidden_layers=16,
        num_attention_heads=4,
        intermediate_size=1536,
        fsmn_kernel_size=11,
        sanm_shift=0,
        lfr_m=7,
        lfr_n=6,
        num_mel_bins=80,
        dropout=0.0,
        blank_id=1024,
        initializer_range=0.02,
        pad_token_id=1024,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.fsmn_kernel_size = fsmn_kernel_size
        self.sanm_shift = sanm_shift
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.num_mel_bins = num_mel_bins
        self.dropout = dropout
        self.blank_id = blank_id
        self.initializer_range = initializer_range
        super().__init__(pad_token_id=pad_token_id, **kwargs)


def apply_lfr(x: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    """批量 LFR(vendor 自 funasr apply_lfr,向量化)。

    左侧重复首帧 (m-1)//2,T' = ceil(T/n)。批内右侧不足补零(原版逐条重复
    末帧;批量下 pad 区已被 mask 归零,偏差只落在每样本最后一个 LFR 窗)。
    """
    b, t, d = x.shape
    t_lfr = math.ceil(t / lfr_n)
    left = x[:, :1].expand(b, (lfr_m - 1) // 2, d)
    x = torch.cat([left, x], dim=1)
    need = (t_lfr - 1) * lfr_n + lfr_m
    if x.size(1) < need:
        x = torch.cat([x, x.new_zeros(b, need - x.size(1), d)], dim=1)
    windows = x.unfold(1, lfr_m, lfr_n)  # (B, T', D, m)
    return windows.transpose(2, 3).reshape(b, t_lfr, d * lfr_m)


class SinusoidalPositionEncoder(nn.Module):
    """funasr 公式:position 从 1 起,inv_timescale 分母 d/2 - 1。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, dim = x.shape
        positions = torch.arange(1, t + 1, dtype=x.dtype, device=x.device)
        log_inc = math.log(10000.0) / (dim / 2 - 1)
        inv = torch.exp(
            torch.arange(dim // 2, dtype=x.dtype, device=x.device) * -log_inc
        )
        scaled = positions[:, None] * inv[None, :]
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)  # (T, dim)
        return x + pe[None]


class SANMAttention(nn.Module):
    """SAN-M:多头自注意力 + FSMN 记忆块(depthwise Conv1d 于 v,残差),相加融合。"""

    def __init__(self, in_feat, n_feat, n_head, kernel_size, sanm_shift, dropout):
        super().__init__()
        assert n_feat % n_head == 0
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_q_k_v = nn.Linear(in_feat, 3 * n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left = (kernel_size - 1) // 2 + sanm_shift
        self.pad_fn = nn.ConstantPad1d((left, kernel_size - 1 - left), 0.0)
        self.dropout = nn.Dropout(dropout)

    def _fsmn(self, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B, T, 1)。depthwise 卷积跨时间,卷积前后都要归零 pad 区防泄漏。
        v = v * mask
        x = self.fsmn_block(self.pad_fn(v.transpose(1, 2))).transpose(1, 2)
        x = self.dropout(x + v)
        return x * mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = torch.split(self.linear_q_k_v(x), self.h * self.d_k, dim=-1)
        fsmn_out = self._fsmn(v, mask)
        q = q.view(b, t, self.h, self.d_k).transpose(1, 2) * self.d_k**-0.5
        k = k.view(b, t, self.h, self.d_k).transpose(1, 2)
        v_h = v.view(b, t, self.h, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)  # (B, h, T, T)
        key_pad = mask.squeeze(-1)[:, None, None, :].eq(0)  # 只 mask key 维
        scores = scores.masked_fill(key_pad, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1).masked_fill(key_pad, 0.0)
        ctx = (self.dropout(attn) @ v_h).transpose(1, 2).reshape(b, t, -1)
        return self.linear_out(ctx) + fsmn_out


class SANMEncoderLayer(nn.Module):
    """pre-norm;in_size != size 时 attention 路径无残差(首层 560→d 投影)。"""

    def __init__(self, in_size, size, n_head, ffn_size, kernel_size, sanm_shift, dropout):
        super().__init__()
        self.in_size = in_size
        self.size = size
        self.norm1 = nn.LayerNorm(in_size)
        self.self_attn = SANMAttention(in_size, size, n_head, kernel_size, sanm_shift, dropout)
        self.norm2 = nn.LayerNorm(size)
        self.ffn = nn.Sequential(
            nn.Linear(size, ffn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        residual = x
        y = self.dropout(self.self_attn(self.norm1(x), mask))
        x = residual + y if self.in_size == self.size else y
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, mask


class SenseVoiceForCTC(PreTrainedModel):
    config_class = SenseVoiceConfig
    main_input_name = "input_features"

    def __init__(self, config: SenseVoiceConfig):
        super().__init__(config)
        d = config.hidden_size
        in_dim = config.num_mel_bins * config.lfr_m
        common = dict(
            n_head=config.num_attention_heads,
            ffn_size=config.intermediate_size,
            kernel_size=config.fsmn_kernel_size,
            sanm_shift=config.sanm_shift,
            dropout=config.dropout,
        )
        self.embed = SinusoidalPositionEncoder()
        self.encoders0 = nn.ModuleList([SANMEncoderLayer(in_dim, d, **common)])
        self.encoders = nn.ModuleList(
            [SANMEncoderLayer(d, d, **common) for _ in range(config.num_hidden_layers - 1)]
        )
        self.after_norm = nn.LayerNorm(d)
        self.ctc_head = nn.Linear(d, config.vocab_size)
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        b, t, _ = input_features.shape
        if attention_mask is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=input_features.device)
        else:
            lengths = attention_mask.sum(-1).long()
        x = apply_lfr(input_features, self.config.lfr_m, self.config.lfr_n)
        out_lengths = (lengths + self.config.lfr_n - 1) // self.config.lfr_n  # ceil
        t_lfr = x.size(1)
        mask = (
            (torch.arange(t_lfr, device=x.device)[None, :] < out_lengths[:, None])
            .to(x.dtype)
            .unsqueeze(-1)
        )  # (B, T', 1)
        x = x * self.config.hidden_size**0.5
        x = self.embed(x)
        for layer in self.encoders0:
            x, mask = layer(x, mask)
        for layer in self.encoders:
            x, mask = layer(x, mask)
        x = self.after_norm(x)
        logits = self.ctc_head(x)
        loss = None
        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, V)
            target_mask = labels != self.config.blank_id
            target_lengths = target_mask.sum(-1)
            loss = F.ctc_loss(
                log_probs,
                labels.masked_fill(~target_mask, 0),
                out_lengths,
                target_lengths,
                blank=self.config.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
        return CausalLMOutput(loss=loss, logits=logits)


def build_tokenizer() -> ParakeetTokenizerFast:
    return ParakeetTokenizerFast.from_pretrained(TOKENIZER_SOURCE)


def build_feature_extractor() -> ParakeetFeatureExtractor:
    return ParakeetFeatureExtractor.from_pretrained(TOKENIZER_SOURCE)


def build_processor(cfg: dict) -> CTCProcessorBundle:
    return CTCProcessorBundle(
        tokenizer=build_tokenizer(),
        feature_extractor=build_feature_extractor(),
    )


def build_model(cfg: dict) -> SenseVoiceForCTC:
    config = SenseVoiceConfig(**{**SMALL_MODEL, **cfg.get("model", {})})
    return SenseVoiceForCTC(config)


def save_checkpoint(model, processor, out_dir: str) -> None:
    model.save_pretrained(out_dir)
    processor.feature_extractor.save_pretrained(out_dir)
    processor.tokenizer.save_pretrained(out_dir)


def load_checkpoint(cfg: dict, ckpt_dir: str) -> tuple:
    model = SenseVoiceForCTC.from_pretrained(ckpt_dir)
    processor = CTCProcessorBundle(
        tokenizer=ParakeetTokenizerFast.from_pretrained(ckpt_dir),
        feature_extractor=ParakeetFeatureExtractor.from_pretrained(ckpt_dir),
    )
    return model, processor


def init_report(model: torch.nn.Module) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    weights = [p for p in model.parameters() if p.dim() >= 2]
    return {
        "params_total": n_params,
        "params_trainable": n_trainable,
        "frozen": frozen,
        "sample_std": weights[len(weights) // 2].std().item(),
    }


if __name__ == "__main__":
    model = build_model({})
    for k, v in init_report(model).items():
        print(f"{k}: {v}")
    cfgm = model.config
    print(f"encoder: d={cfgm.hidden_size} layers={cfgm.num_hidden_layers} lfr=({cfgm.lfr_m},{cfgm.lfr_n})")
