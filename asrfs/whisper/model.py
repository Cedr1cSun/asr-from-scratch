from pathlib import Path

import torch
import yaml
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── 适配契约常量(冻结)──────────────────────────────────────────────
LOSS_FAMILY = "ce"
LABEL_PAD_ID = -100
EXPECTED_FROZEN = {"model.encoder.embed_positions.weight"}

_COMMON = dict(
    vocab_size=51864,
    num_mel_bins=80,
    max_source_positions=1500,
    max_target_positions=448,
    activation_function="gelu",
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    init_std=0.02,
    bos_token_id=50257,
    eos_token_id=50256,
    pad_token_id=50256,
    decoder_start_token_id=50257,
)

SIZE_PRESETS = {
    "small": dict(
        _COMMON,
        d_model=768,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_layers=12,
        decoder_attention_heads=12,
        decoder_ffn_dim=3072,
    ),
    "medium": dict(
        _COMMON,
        d_model=1024,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_layers=24,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
    ),
}


def tokenizer_source(size: str) -> str:
    return f"openai/whisper-{size}.en"


# pin revision:tokenizer/FE 决定 label/特征字节,Hub 侧更新不能悄悄改字节
# (round-2 审计 F3);指纹入 full_data.params_hash。未列 size 不 pin(revision=None
# 等价省略),指纹标 unpinned。
TOKENIZER_REVISIONS = {
    "medium": "2e98eb6279edf5095af0c8dedb36bdec0acd172b",
    "small": "e8727524f962ee844a7319d92be39ac1bd25655a",
}


def tokenizer_fingerprint(cfg: dict) -> str:
    """标签/特征字节身份,入 full_data.params_hash(round-2 审计 F1/F2)。"""
    size = cfg["model"]["size"]
    return f"{tokenizer_source(size)}@{TOKENIZER_REVISIONS.get(size, 'unpinned')}"


def build_processor(cfg: dict) -> WhisperProcessor:
    """适配契约 build_processor:统一收 cfg。"""
    size = cfg["model"]["size"]
    return WhisperProcessor.from_pretrained(
        tokenizer_source(size), revision=TOKENIZER_REVISIONS.get(size)
    )


def build_model(cfg: dict) -> WhisperForConditionalGeneration:
    """适配契约 build_model:全参数随机初始化;仅 generation_config 等非权重配置从 Hub 拉。"""
    m = cfg["model"]
    config = WhisperConfig(**SIZE_PRESETS[m["size"]], apply_spec_augment=m["apply_spec_augment"])
    model = WhisperForConditionalGeneration(config)
    model.generation_config = GenerationConfig.from_pretrained(
        tokenizer_source(m["size"]), revision=TOKENIZER_REVISIONS.get(m["size"])
    )
    # decode 契约的解码长度单一来源:cfg model.generation_max_length -> generation_config
    model.generation_config.max_new_tokens = int(m["generation_max_length"])
    return model


def init_report(model: torch.nn.Module) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    enc_attn = model.model.encoder.layers[0].self_attn.q_proj.weight
    dec_attn = model.model.decoder.layers[0].self_attn.q_proj.weight
    return {
        "params_total": n_params,
        "params_trainable": n_trainable,
        "frozen": frozen,
        "enc_l0_q_std": enc_attn.std().item(),
        "dec_l0_q_std": dec_attn.std().item(),
    }


def decode(model, processor, batch) -> list:
    """适配契约 decode:generate + batch_decode;容忍 batch 携带 labels 等训练键并忽略。"""
    input_features = batch["input_features"]  # 只取声学输入,labels 等其余键就地忽略
    input_features = input_features.to(next(model.parameters()).device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred_ids = model.generate(
            input_features, max_new_tokens=model.generation_config.max_new_tokens
        )
    if was_training:
        model.train()
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)


def save_checkpoint(model, processor, out_dir: str) -> None:
    """适配契约 save_checkpoint:标准 save_pretrained 目录(SURE-EVAL 以 MODEL_PATH 指向即可加载)。"""
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def load_checkpoint(cfg: dict, ckpt_dir: str) -> tuple:
    """适配契约 load_checkpoint。cfg 为契约签名要求;whisper 侧加载不需读它。"""
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir)
    processor = WhisperProcessor.from_pretrained(ckpt_dir)
    return model, processor


if __name__ == "__main__":
    cfg = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())
    model = build_model(cfg)
    report = init_report(model)
    for k, v in report.items():
        print(f"{k}: {v}")
    assert report["frozen"] == EXPECTED_FROZEN, f"unexpected frozen params: {report['frozen']}"
    assert 0.01 < report["enc_l0_q_std"] < 0.03, "init std should be ~0.02"
    print("model self-check OK")
