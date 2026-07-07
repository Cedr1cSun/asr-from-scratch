from pathlib import Path

import pytest
import yaml

from asrfs.common import full_data

REPO = Path(__file__).resolve().parent.parent
MODELS = ["whisper", "parakeet", "sensevoice", "x_asr"]


def _load(model: str, name: str) -> dict:
    return yaml.safe_load((REPO / "asrfs" / model / name).read_text())


@pytest.mark.parametrize("model", MODELS)
def test_hash_equality_config_vs_config_full(model, monkeypatch):
    """防 drift(spec §3):params_hash 覆盖键两文件必须同值。config_full 走 manifest
    源、config.yaml 走 hf 源,数据集身份键有意分叉(manifest-loader spec §三),故
    强制 hf 口径比等——只锁模型/过滤键不漂;manifest 键值本身由
    test_config_full_manifest_source 锁。"""
    monkeypatch.setenv("ASRFS_DATA_SOURCE", "hf")
    cfg_full = _load(model, "config_full.yaml")
    cfg_smoke = _load(model, "config.yaml")
    assert full_data.params_hash(
        cfg_full, tokenizer_fingerprint=full_data._tokenizer_fingerprint(model, cfg_full)
    ) == full_data.params_hash(
        cfg_smoke, tokenizer_fingerprint=full_data._tokenizer_fingerprint(model, cfg_smoke)
    )


@pytest.mark.parametrize("model", MODELS)
def test_config_full_augment_section(model):
    sa = _load(model, "config_full.yaml")["augment"]["spec_augment"]
    assert sa["time_axis"] == (1 if model == "whisper" else 0)
    assert (sa["num_feature_masks"], sa["features_mask_size"]) == (2, 27)
    assert (sa["num_frame_masks"], sa["frames_mask_size"]) == (10, 100)
    assert sa["max_frames_mask_fraction"] == 0.15 and sa["p"] == 0.9


@pytest.mark.parametrize(
    "model,max_steps,warmup,probe_bs,ga,n_gpu",
    [("whisper", 33000, 4000, 4, 16, 4), ("parakeet", 66000, 2000, 32, 4, 1),
     ("sensevoice", 66000, 2000, 32, 4, 1), ("x_asr", 66000, 50, 8, 8, 2)],
)
def test_config_full_960h_training_values(model, max_steps, warmup, probe_bs, ga, n_gpu):
    """2026-07-07 3090 标定:per_device=batch_probe 实测,GA 按排卡(whisper 4 卡/
    parakeet 1/sensevoice 1/x_asr 2)配平;epoch 按原始 281241 行计、parakeet/
    sensevoice 50→30ep(用户 2026-07-08 定 3-5 天时限)。有效 batch 等式含 n_gpu,
    单看 config 两键不再自洽。"""
    t = _load(model, "config_full.yaml")["training"]
    assert t["max_steps"] == max_steps and t["warmup_steps"] == warmup
    assert t["eval_steps"] == 1000 and t["save_steps"] == 1000
    assert t["per_device_train_batch_size"] == probe_bs
    assert t["gradient_accumulation_steps"] == ga
    eff = probe_bs * ga * n_gpu
    assert eff == (256 if model == "whisper" else 128)


def test_whisper_full_disables_generate_eval():
    t = _load("whisper", "config_full.yaml")["training"]
    assert t["predict_with_generate"] is False


def test_x_asr_full_eden_keys():
    t = _load("x_asr", "config_full.yaml")["training"]
    assert t["lr_batches"] == 7500 and t["lr_epochs"] == 3.5


@pytest.mark.parametrize("model", MODELS)
def test_config_full_manifest_source(model):
    """公司数据线(manifest-loader spec §二):四份 config_full 钉死 manifest 源与集群路径。"""
    d = _load(model, "config_full.yaml")["data"]
    assert d["source"] == "manifest"
    assert d["manifest_path"] == "/hpc_stor03/sjtu_home/ruichen.sun/librispeech_train_960.jsonl"
