import torch
from transformers import (
    ParakeetCTCConfig,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetForCTC,
    ParakeetTokenizerFast,
)

TOKENIZER_SOURCE = "nvidia/parakeet-ctc-0.6b"

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

def build_tokenizer() -> ParakeetTokenizerFast:
    return ParakeetTokenizerFast.from_pretrained(TOKENIZER_SOURCE)

def build_feature_extractor() -> ParakeetFeatureExtractor:
    return ParakeetFeatureExtractor.from_pretrained(TOKENIZER_SOURCE)

def build_model() -> ParakeetForCTC:
    tokenizer = build_tokenizer()
    encoder_config = ParakeetEncoderConfig(**SMALL_ENCODER)
    config = ParakeetCTCConfig(
        encoder_config=encoder_config.to_dict(),
        vocab_size=tokenizer.vocab_size + 1,
        pad_token_id=tokenizer.vocab_size,
    )
    return ParakeetForCTC(config)

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
    model = build_model()
    for k, v in init_report(model).items():
        print(f"{k}: {v}")
    enc = model.config.encoder_config
    print(f"encoder: d={enc.hidden_size} layers={enc.num_hidden_layers} subsampling={enc.subsampling_factor}")
