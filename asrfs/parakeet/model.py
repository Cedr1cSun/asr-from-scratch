from dataclasses import dataclass

import torch
from transformers import (
    ParakeetCTCConfig,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetForCTC,
    ParakeetTokenizerFast,
)

TOKENIZER_SOURCE = "nvidia/parakeet-ctc-0.6b"
# pin revision:tokenizer/FE 决定 label/特征字节,Hub 侧更新不能悄悄改字节
# (round-2 审计 F3);指纹入 full_data.params_hash。
TOKENIZER_REVISION = "ad09ba1cc62743fbc9814de5d2016fca9096485a"

LOSS_FAMILY = "ctc"
# LABEL_PAD_ID 必须等于 CTC blank id(= tokenizer.vocab_size)。tokenizer 来自
# Hub(TOKENIZER_SOURCE),import 时无网络/无缓存即无法解析真值,故硬编码
# nvidia/parakeet-ctc-0.6b 的已知值 1024。守卫:
# tests/test_parakeet_adapter.py::test_label_pad_id_matches_tokenizer(slow)
# 断言 LABEL_PAD_ID == build_tokenizer().vocab_size,tokenizer 源变更即报警。
LABEL_PAD_ID = 1024
EXPECTED_FROZEN: set = set()

SMALL_ENCODER = dict(
    hidden_size=256,
    num_hidden_layers=16,
    num_attention_heads=4,
    intermediate_size=1024,
    subsampling_factor=8,
    subsampling_conv_channels=256,
    conv_kernel_size=9,
    num_mel_bins=80,
    dropout=0.0,
)


@dataclass
class ParakeetProcessorBundle:
    tokenizer: ParakeetTokenizerFast
    feature_extractor: ParakeetFeatureExtractor


def build_tokenizer() -> ParakeetTokenizerFast:
    return ParakeetTokenizerFast.from_pretrained(TOKENIZER_SOURCE, revision=TOKENIZER_REVISION)


def tokenizer_fingerprint(cfg: dict) -> str:
    """标签/特征字节身份,入 full_data.params_hash(round-2 审计 F1/F2)。"""
    return f"{TOKENIZER_SOURCE}@{TOKENIZER_REVISION}"


def build_feature_extractor() -> ParakeetFeatureExtractor:
    return ParakeetFeatureExtractor.from_pretrained(TOKENIZER_SOURCE, revision=TOKENIZER_REVISION)


def build_processor(cfg: dict) -> ParakeetProcessorBundle:
    return ParakeetProcessorBundle(
        tokenizer=build_tokenizer(),
        feature_extractor=build_feature_extractor(),
    )


def build_model(cfg: dict) -> ParakeetForCTC:
    tokenizer = build_tokenizer()
    encoder_config = ParakeetEncoderConfig(**{**SMALL_ENCODER, **cfg.get("model", {})})
    config = ParakeetCTCConfig(
        encoder_config=encoder_config.to_dict(),
        vocab_size=tokenizer.vocab_size + 1,
        pad_token_id=tokenizer.vocab_size,
    )
    return ParakeetForCTC(config)


def save_checkpoint(model, processor, out_dir: str) -> None:
    model.save_pretrained(out_dir)
    processor.feature_extractor.save_pretrained(out_dir)
    processor.tokenizer.save_pretrained(out_dir)


def load_checkpoint(cfg: dict, ckpt_dir: str) -> tuple:
    model = ParakeetForCTC.from_pretrained(ckpt_dir)
    processor = ParakeetProcessorBundle(
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
    enc = model.config.encoder_config
    print(f"encoder: d={enc.hidden_size} layers={enc.num_hidden_layers} subsampling={enc.subsampling_factor}")
