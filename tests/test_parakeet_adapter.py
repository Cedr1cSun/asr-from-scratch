from pathlib import Path

import pytest
import yaml

from asrfs.parakeet import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    ParakeetProcessorBundle,
    build_model,
    build_processor,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "asrfs" / "parakeet" / "config.yaml"

TINY_MODEL_CFG = {
    "model": {
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "subsampling_conv_channels": 64,
    }
}


@pytest.fixture(scope="module")
def processor_bundle():
    return build_processor({})


def test_constants():
    assert LOSS_FAMILY == "ctc"
    assert isinstance(LABEL_PAD_ID, int) and LABEL_PAD_ID == 1024
    assert isinstance(EXPECTED_FROZEN, set) and EXPECTED_FROZEN == set()


def test_config_yaml_harness_keys():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    t = cfg["training"]
    for key in (
        "learning_rate",
        "warmup_steps",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "seed",
    ):
        assert key in t, f"training.{key} missing"
    assert t["seed"] == 42
    assert cfg["data"]["n_train"] == 100 and cfg["data"]["n_eval"] == 20
    s = cfg["smoke"]
    assert s["overfit1_steps"] == 500
    assert s["overfit1_lr"] == pytest.approx(3.0e-4)
    assert s["mini100_steps"] == 250
    assert s["mini100_lr"] == pytest.approx(3.0e-4)


@pytest.mark.slow
def test_label_pad_id_matches_tokenizer():
    # LABEL_PAD_ID 是硬编码值,此测试钉住它 == 真实 tokenizer 的 vocab_size,
    # tokenizer 源(nvidia/parakeet-ctc-0.6b)变更时在此报警。
    from asrfs.parakeet.model import build_tokenizer

    assert LABEL_PAD_ID == build_tokenizer().vocab_size


@pytest.mark.slow
def test_build_processor_bundle(processor_bundle):
    assert isinstance(processor_bundle, ParakeetProcessorBundle)
    assert processor_bundle.tokenizer.vocab_size == LABEL_PAD_ID
    assert processor_bundle.feature_extractor.sampling_rate == 16000


@pytest.mark.slow
def test_build_model_cfg_overrides():
    tiny = build_model(TINY_MODEL_CFG)
    assert tiny.config.encoder_config.hidden_size == 64
    assert tiny.config.encoder_config.num_hidden_layers == 1
    assert tiny.config.vocab_size == LABEL_PAD_ID + 1
    assert tiny.config.pad_token_id == LABEL_PAD_ID
    default = build_model({})
    assert default.config.encoder_config.hidden_size == 256
    assert default.config.encoder_config.num_hidden_layers == 16
