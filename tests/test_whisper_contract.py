import inspect
from pathlib import Path

import pytest
import yaml

import asrfs.whisper as whisper
from asrfs.whisper.model import init_report

CONFIG_PATH = Path(__file__).resolve().parents[1] / "asrfs" / "whisper" / "config.yaml"

# 契约签名(冻结)。A4/A5 落地后本 dict 逐步补齐到 9 函数。
# 前缀匹配语义与 harness check_contract 一致:必需参数名按序前缀匹配,允许尾部额外带默认值参数。
EXPECTED_SIGNATURES = {
    "build_processor": ["cfg"],
    "build_model": ["cfg"],
}


def test_constants_present_and_typed():
    assert whisper.LOSS_FAMILY == "ce"
    assert isinstance(whisper.LOSS_FAMILY, str)
    assert whisper.LABEL_PAD_ID == -100
    assert isinstance(whisper.LABEL_PAD_ID, int)
    assert whisper.EXPECTED_FROZEN == {"model.encoder.embed_positions.weight"}
    assert isinstance(whisper.EXPECTED_FROZEN, set)


def test_contract_signatures():
    for name, params in EXPECTED_SIGNATURES.items():
        fn = getattr(whisper, name)
        got = [p.name for p in inspect.signature(fn).parameters.values()]
        assert got[: len(params)] == params, f"{name}: {got} != {params}"


def test_config_yaml_harness_keys():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    for key in ["learning_rate", "warmup_steps", "max_steps", "per_device_train_batch_size",
                "gradient_accumulation_steps", "logging_steps", "eval_steps", "seed"]:
        assert key in cfg["training"], key
    assert cfg["training"]["seed"] == 42
    for key in ["n_train", "n_eval"]:
        assert key in cfg["data"], key
    for key in ["overfit1_steps", "overfit1_lr", "mini100_steps", "mini100_lr"]:
        assert key in cfg["smoke"], key
    # 20 epoch 等效:20 * n_train / (bs * grad_accum) = 20 * 100 / (2 * 4) = 250
    assert cfg["smoke"]["mini100_steps"] == 250
    for key in ["size", "apply_spec_augment", "gradient_checkpointing", "generation_max_length"]:
        assert key in cfg["model"], key


@pytest.mark.slow
def test_build_model_from_cfg():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model = whisper.build_model(cfg)
    report = init_report(model)
    assert 750e6 < report["params_total"] < 790e6
    assert report["frozen"] == whisper.EXPECTED_FROZEN
    assert 0.01 < report["enc_l0_q_std"] < 0.03
