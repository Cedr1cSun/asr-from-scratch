"""X-ASR(icefall zipformer pruned-RNNT 配方的 HF 包装,标准无剪枝 RNN-T loss)。

encoder/decoder/joiner/优化器全部来自 _vendor(icefall @ 7a35ca2,手术清单见
_vendor/VENDOR.md);本文件只做:HF Config/PreTrainedModel 包装、forward 的
RNN-T 接线(全笛卡尔 joint + torchaudio rnnt_loss)、greedy 解码、契约函数。
结构与决策见 docs/superpowers/specs/2026-07-03-x-asr-from-scratch-design.md。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import torchaudio
from transformers import (
    ParakeetFeatureExtractor,
    ParakeetTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

from asrfs.common.ctc import CTCProcessorBundle
from asrfs.x_asr._vendor.decoder import Decoder
from asrfs.x_asr._vendor.icefall_compat import make_pad_mask
from asrfs.x_asr._vendor.joiner import Joiner
from asrfs.x_asr._vendor.subsampling import Conv2dSubsampling
from asrfs.x_asr._vendor.zipformer import Zipformer2

TOKENIZER_SOURCE = "nvidia/parakeet-ctc-0.6b"

_BPE_MODEL = Path(__file__).resolve().parent / "bpe" / "librispeech_bpe500.model"

LOSS_FAMILY = "rnnt"
# LABEL_PAD_ID = blank = SentencePiece vocab_size(同 CTC 族;RNN-T 里 blank 兼任 SOS)。
# 守卫:tests/test_xasr_adapter.py::test_label_pad_id_matches_tokenizer(slow)。
LABEL_PAD_ID = 500
EXPECTED_FROZEN: set = set()


class SpmTokenizer:
    """SentencePiece 薄 wrapper,接口对齐 common/ctc.py 的现用法
    (tokenizer(text)["input_ids"] / tokenizer.decode(ids))。"""

    def __init__(self, model_path: Path = _BPE_MODEL):
        self._sp = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()  # 500

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self._sp.encode(text, out_type=int)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._sp.decode(list(ids))


# Zipformer-S 缩小预设(spec §5;接口文档 §a 的 train.py 映射,cnn_module_kernel 取 argparse 缺省)
SMALL_MODEL = dict(
    num_mel_bins=80,
    downsampling_factor=(1, 2, 4, 8, 4, 2),
    num_encoder_layers=(2, 2, 2, 2, 2, 2),
    encoder_dim=(192, 256, 256, 256, 256, 256),
    encoder_unmasked_dim=(192, 192, 192, 192, 192, 192),
    query_head_dim=32,
    pos_head_dim=4,
    value_head_dim=12,
    pos_dim=48,
    num_heads=(4, 4, 4, 8, 4, 4),
    feedforward_dim=(512, 768, 768, 768, 768, 768),
    cnn_module_kernel=(31, 31, 15, 15, 15, 31),
    decoder_dim=512,
    joiner_dim=512,
    context_size=2,
    dropout=0.0,
)


class XASRConfig(PretrainedConfig):
    model_type = "x-asr-rnnt"

    def __init__(
        self,
        vocab_size=501,
        blank_id=500,
        num_mel_bins=80,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        num_encoder_layers=(2, 2, 2, 2, 2, 2),
        encoder_dim=(192, 256, 256, 256, 256, 256),
        encoder_unmasked_dim=(192, 192, 192, 192, 192, 192),
        query_head_dim=32,
        pos_head_dim=4,
        value_head_dim=12,
        pos_dim=48,
        num_heads=(4, 4, 4, 8, 4, 4),
        feedforward_dim=(512, 768, 768, 768, 768, 768),
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        decoder_dim=512,
        joiner_dim=512,
        context_size=2,
        dropout=0.0,
        causal=False,
        pad_token_id=500,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_mel_bins = num_mel_bins
        self.downsampling_factor = tuple(downsampling_factor)
        self.num_encoder_layers = tuple(num_encoder_layers)
        self.encoder_dim = tuple(encoder_dim)
        self.encoder_unmasked_dim = tuple(encoder_unmasked_dim)
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.value_head_dim = value_head_dim
        self.pos_dim = pos_dim
        self.num_heads = tuple(num_heads)
        self.feedforward_dim = tuple(feedforward_dim)
        self.cnn_module_kernel = tuple(cnn_module_kernel)
        self.decoder_dim = decoder_dim
        self.joiner_dim = joiner_dim
        self.context_size = context_size
        self.dropout = dropout
        self.causal = causal
        super().__init__(pad_token_id=pad_token_id, **kwargs)


@dataclass
class XASRModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    encoder_out: Optional[torch.FloatTensor] = None
    encoder_out_lens: Optional[torch.LongTensor] = None


class XASRForRNNT(PreTrainedModel):
    config_class = XASRConfig
    main_input_name = "input_features"
    # transformers 5.12 Trainer:forward 带 **kwargs(VAR_KEYWORD)会被判定为
    # model_accepts_loss_kwargs=True,从而跳过 training_step 里
    # loss/gradient_accumulation_steps 的归一化;我们的 forward 不消费
    # num_items_in_batch,grad_accum>1 时梯度被放大 G 倍。显式声明 False(Trainer
    # 在 __init__ 读该类属性,见 trainer.py model_accepts_loss_kwargs 分支)。
    accepts_loss_kwargs = False

    def __init__(self, config: XASRConfig):
        super().__init__(config)
        self.encoder_embed = Conv2dSubsampling(
            in_channels=config.num_mel_bins,
            out_channels=config.encoder_dim[0],
            dropout=config.dropout,
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=config.downsampling_factor,
            num_encoder_layers=config.num_encoder_layers,
            encoder_dim=config.encoder_dim,
            encoder_unmasked_dim=config.encoder_unmasked_dim,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            pos_dim=config.pos_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            dropout=config.dropout,
            warmup_batches=4000.0,
            causal=config.causal,
            chunk_size=(-1,),
            left_context_frames=(-1,),
        )
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            decoder_dim=config.decoder_dim,
            blank_id=config.blank_id,
            context_size=config.context_size,
        )
        self.joiner = Joiner(
            encoder_dim=max(config.encoder_dim),
            decoder_dim=config.decoder_dim,
            joiner_dim=config.joiner_dim,
            vocab_size=config.vocab_size,
        )
        self.post_init()

    def _init_weights(self, module):
        # 忠实度红线:icefall 模块自带初始化(ScaledLinear initial_scale 等),
        # 一律不覆写。no-op 同时保证 from_pretrained 不重随机化(fast-init 守卫)。
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model._reset_meta_pos_encodings()
        return model

    def _reset_meta_pos_encodings(self):
        # transformers 5.12 fast-init 在 meta 设备上构造子模块;vendored
        # CompactRelPositionalEncoding.pe 是惰性普通属性(非 buffer/参数,不在
        # state_dict),于是停留在 meta 且不会被权重加载物化。清成 None,让下次
        # forward 在真实设备上按同一确定性公式重算(值与新建模型逐位一致)。
        for m in self.modules():
            pe = getattr(m, "pe", None)
            if isinstance(pe, torch.Tensor) and pe.is_meta:
                m.pe = None

    def encode(self, input_features, attention_mask=None):
        """(B,T,80) @100Hz -> (B,T2,C) @~25Hz;C = max(encoder_dim)。"""
        b, t, _ = input_features.shape
        if attention_mask is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=input_features.device)
        else:
            lengths = attention_mask.sum(-1).long()
        x, x_lens = self.encoder_embed(input_features, lengths)  # (B,T1,D0), (B,)
        mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # -> (T1,B,D0) time-major
        encoder_out, out_lens = self.encoder(x, x_lens, mask)
        return encoder_out.permute(1, 0, 2), out_lens  # (B,T2,C)

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        encoder_out, encoder_out_lens = self.encode(input_features, attention_mask)
        loss = None
        if labels is not None:
            blank = self.config.blank_id
            target_mask = labels != blank
            target_lengths = target_mask.sum(-1)
            targets = labels.masked_fill(~target_mask, 0)
            # torchaudio rnnt_loss 契约:logits 的 U 维必须 == max(target_lengths)+1,
            # targets 宽度 == max(target_lengths)(不是 labels 的填充宽度)。实测:U 维
            # 超出会 RuntimeError "output length mismatch"。裁到最长真实长度;正常训练下
            # collator 已把最长行填满,此裁剪为 no-op,仅对超填充的手构 batch 生效。
            max_u = int(target_lengths.max().item())
            targets = targets[:, :max_u]
            sos = torch.full(
                (labels.size(0), 1), blank, dtype=labels.dtype, device=labels.device
            )
            sos_y = torch.cat([sos, targets], dim=1)  # (B, U+1),blank 兼任 SOS
            decoder_out = self.decoder(sos_y, need_pad=True)  # (B, U+1, decoder_dim)
            logits = self.joiner(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1), project_input=True
            )  # (B, T2, U+1, V) —— 全笛卡尔,无剪枝
            loss = torchaudio.functional.rnnt_loss(
                logits=logits.float(),
                targets=targets.int(),
                # 每样本真实编码器帧数(encode 返回的 out_lens),不是 batch 的 padded
                # 时间维。用 encoder_out.size(1) 会把 padding 帧塞进短样本的 RNN-T 网格,
                # reduction="mean" 下短样本 loss 被污染(实测 batch != 单样本均值)。
                logit_lengths=encoder_out_lens.int(),
                target_lengths=target_lengths.int(),
                blank=blank,
                reduction="mean",
            )
        return XASRModelOutput(
            loss=loss, encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
        )

    @torch.no_grad()
    def greedy_decode(
        self, input_features, attention_mask=None, max_sym_per_frame: int = 32
    ) -> list:
        """标准 transducer greedy:逐帧 emit-until-blank,安全帽防未训练模型死循环;
        msf=1 简化被 overfit1 判据证伪,见 harness 标定记录。"""
        was_training = self.training
        self.eval()
        try:
            encoder_out, lens = self.encode(input_features, attention_mask)
            blank, ctx = self.config.blank_id, self.config.context_size
            hyps = []
            for i in range(encoder_out.size(0)):
                # icefall greedy_search 的种子:[-1]*(context_size-1)+[blank]。训练时
                # decoder need_pad=True 对 conv 左侧补零,vendored Decoder 用负 id 让
                # 对应 embedding 归零(decoder.py:115-117)以在推理端逐位复刻位置 0 的
                # predictor 状态。用 [blank]*ctx 会偏离训练首帧状态(实测差 ~1.3)。
                hyp = [-1] * (ctx - 1) + [blank]
                dec_in = torch.tensor([hyp[-ctx:]], device=encoder_out.device)
                dec_out = self.decoder(dec_in, need_pad=False)  # (1,1,D)
                tokens = []
                for t in range(int(lens[i].item())):
                    # 标准 transducer greedy:本帧反复发射直到 joiner argmax==blank
                    # 才前进到下一帧(Graves 2012;icefall greedy_search max_sym>1;
                    # torchaudio)。max_sym_per_frame 是安全帽:未训练模型近似均匀发射
                    # (blank 概率 ~1/vocab_size),几乎每步都非 blank,无帽循环在随机
                    # 权重上会 T2×∞ 空转。32/帧把最坏情况锁到 32·T2 次微型 decoder
                    # 调用,又远高于任何合理的 burst 发射密度。
                    n_emit = 0
                    while n_emit < max_sym_per_frame:
                        logit = self.joiner(
                            encoder_out[i : i + 1, t : t + 1].unsqueeze(2),
                            dec_out.unsqueeze(1),
                            project_input=True,
                        )  # (1,1,1,V)
                        tok = int(logit.reshape(-1).argmax().item())
                        if tok == blank:
                            break
                        tokens.append(tok)
                        hyp.append(tok)
                        dec_in = torch.tensor([hyp[-ctx:]], device=encoder_out.device)
                        dec_out = self.decoder(dec_in, need_pad=False)
                        n_emit += 1
                hyps.append(tokens)
            return hyps
        finally:
            if was_training:
                self.train()


def build_tokenizer() -> SpmTokenizer:
    return SpmTokenizer()


def build_feature_extractor() -> ParakeetFeatureExtractor:
    return ParakeetFeatureExtractor.from_pretrained(TOKENIZER_SOURCE)


def build_processor(cfg: dict) -> CTCProcessorBundle:
    return CTCProcessorBundle(
        tokenizer=build_tokenizer(),
        feature_extractor=build_feature_extractor(),
    )


def build_model(cfg: dict) -> XASRForRNNT:
    config = XASRConfig(**{**SMALL_MODEL, **cfg.get("model", {})})
    return XASRForRNNT(config)


def save_checkpoint(model, processor, out_dir: str) -> None:
    model.save_pretrained(out_dir)
    processor.feature_extractor.save_pretrained(out_dir)
    processor.tokenizer.save_pretrained(out_dir)


def load_checkpoint(cfg: dict, ckpt_dir: str) -> tuple:
    model = XASRForRNNT.from_pretrained(ckpt_dir)
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
    c = model.config
    print(f"encoder dims={c.encoder_dim} layers={c.num_encoder_layers} causal={c.causal}")
