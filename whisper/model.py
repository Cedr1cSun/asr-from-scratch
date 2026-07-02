import torch
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

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

def build_processor(size: str = "medium") -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(tokenizer_source(size))

def build_model(size: str = "medium", apply_spec_augment: bool = False) -> WhisperForConditionalGeneration:
    config = WhisperConfig(**SIZE_PRESETS[size], apply_spec_augment=apply_spec_augment)
    model = WhisperForConditionalGeneration(config)
    model.generation_config = GenerationConfig.from_pretrained(tokenizer_source(size))
    return model

EXPECTED_FROZEN = {"model.encoder.embed_positions.weight"}

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

if __name__ == "__main__":
    model = build_model()
    report = init_report(model)
    for k, v in report.items():
        print(f"{k}: {v}")
    assert report["frozen"] == EXPECTED_FROZEN, f"unexpected frozen params: {report['frozen']}"
    assert 0.01 < report["enc_l0_q_std"] < 0.03, "init std should be ~0.02"
    print("model self-check OK")
