from pathlib import Path

import pytest
import yaml

from asrfs.common import full_data

REPO = Path(__file__).resolve().parent.parent
MODELS = ["whisper", "parakeet", "sensevoice", "x_asr"]


def _load(model: str, name: str) -> dict:
    return yaml.safe_load((REPO / "asrfs" / model / name).read_text())


@pytest.mark.parametrize("model", MODELS)
def test_hash_equality_config_vs_config_full(model):
    """防 drift(spec §3):params_hash 覆盖键两文件必须同值。"""
    assert full_data.params_hash(_load(model, "config_full.yaml")) == full_data.params_hash(
        _load(model, "config.yaml")
    )


@pytest.mark.parametrize("model", MODELS)
def test_config_full_augment_section(model):
    sa = _load(model, "config_full.yaml")["augment"]["spec_augment"]
    assert sa["time_axis"] == (1 if model == "whisper" else 0)
    assert (sa["num_feature_masks"], sa["features_mask_size"]) == (2, 27)
    assert (sa["num_frame_masks"], sa["frames_mask_size"]) == (10, 100)
    assert sa["max_frames_mask_fraction"] == 0.15 and sa["p"] == 0.9


@pytest.mark.parametrize(
    "model,max_steps,warmup",
    [("whisper", 33000, 4000), ("parakeet", 110000, 2000),
     ("sensevoice", 110000, 2000), ("x_asr", 66000, 50)],
)
def test_config_full_960h_training_values(model, max_steps, warmup):
    t = _load(model, "config_full.yaml")["training"]
    assert t["max_steps"] == max_steps and t["warmup_steps"] == warmup
    assert t["eval_steps"] == 1000 and t["save_steps"] == 1000
    eff = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    assert eff == (256 if model == "whisper" else 128)


def test_whisper_full_disables_generate_eval():
    t = _load("whisper", "config_full.yaml")["training"]
    assert t["predict_with_generate"] is False


def test_x_asr_full_eden_keys():
    t = _load("x_asr", "config_full.yaml")["training"]
    assert t["lr_batches"] == 7500 and t["lr_epochs"] == 3.5
