"""full_data 单元测试:manifest 写入 + 过滤,全部走 fake 内存行,零网络。

真实 960h 预计算在集群侧跑(磁盘/时长预算见 asrfs/common/full_data.py 模块
docstring);本套件只锁转换、过滤、manifest、加载语义。
"""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from datasets import load_from_disk

from asrfs.common import full_data

CFG = {
    "model_size": "fake",
    "run_name": "full_data_unit",
    "data": {"n_train": 100, "n_eval": 20, "max_label_len": 5, "max_audio_seconds": 30.0},
    "training": {"learning_rate": 1.0e-4, "max_steps": 300},
}

SPLIT_NAMES = ["train.clean.100", "train.clean.360", "train.other.500", "validation.clean"]


class FakeAdapter:
    __name__ = "asrfs.faketest"

    @staticmethod
    def build_processor(cfg):
        return "fake-processor"

    @staticmethod
    def make_example(processor, audio, sampling_rate, text):
        assert processor == "fake-processor"
        frames = max(1, len(audio) // 1600)
        feats = np.full((frames, 2), float(len(audio)), dtype=np.float32)
        return {"input_features": feats, "labels": [ord(c) % 32 for c in text.lower()]}


def _fake_rows(tag):
    sr = 16000
    return [
        {"id": f"{tag}-keep", "audio_array": np.zeros(sr, dtype=np.float32),
         "sampling_rate": sr, "text": "ok"},
        {"id": f"{tag}-audio-too-long", "audio_array": np.zeros(31 * sr, dtype=np.float32),
         "sampling_rate": sr, "text": "ok"},
        {"id": f"{tag}-label-too-long", "audio_array": np.zeros(sr, dtype=np.float32),
         "sampling_rate": sr, "text": "xxxxxxxxx"},
    ]


def _fake_stream(config, split, subset_head=None):
    rows = _fake_rows(f"{config}.{split}")
    if subset_head is not None:
        rows = rows[:subset_head]
    yield from rows


@pytest.fixture()
def data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(full_data, "_stream_split", _fake_stream)
    return tmp_path


def test_prepare_writes_manifest_and_filters(data_root):
    manifest = full_data.prepare_full_dataset(CFG, FakeAdapter)

    assert sorted(manifest["splits"]) == sorted(SPLIT_NAMES)
    for split in SPLIT_NAMES:
        # 3 条 fake 行:1 条保留、1 条 31s 音频过滤、1 条 label 长度 9 > 5 过滤
        assert manifest["splits"][split] == {"rows_before": 3, "rows_after": 1}
    assert manifest["dtype"] == "float16"
    assert manifest["subset_head"] is None
    assert manifest["feature_dir"] == str(data_root / "full" / "faketest")

    on_disk = json.loads((data_root / "full" / "faketest" / "manifest.json").read_text())
    assert on_disk == manifest

    ds = load_from_disk(str(data_root / "full" / "faketest" / "train.clean.100"))
    assert len(ds) == 1
    assert "float16" in str(ds.features["input_features"])
    assert ds[0]["length"] == 16000
    assert ds[0]["labels"] == [ord("o") % 32, ord("k") % 32]


def test_prepare_respects_subset_head(data_root):
    manifest = full_data.prepare_full_dataset(CFG, FakeAdapter, subset_head=1)
    for split in SPLIT_NAMES:
        assert manifest["splits"][split] == {"rows_before": 1, "rows_after": 1}
    assert manifest["subset_head"] == 1


def test_params_hash_covers_feature_params_only():
    base = full_data.params_hash(CFG)

    irrelevant = copy.deepcopy(CFG)
    irrelevant["training"]["learning_rate"] = 9.9e-9
    irrelevant["data"]["n_train"] = 7
    irrelevant["run_name"] = "other_run"
    irrelevant["smoke"] = {"overfit1_steps": 1}
    assert full_data.params_hash(irrelevant) == base

    model_changed = copy.deepcopy(CFG)
    model_changed["model_size"] = "not-fake"
    assert full_data.params_hash(model_changed) != base

    filter_changed = copy.deepcopy(CFG)
    filter_changed["data"]["max_label_len"] = 448
    assert full_data.params_hash(filter_changed) != base


def test_load_full_dataset_roundtrip(data_root):
    full_data.prepare_full_dataset(CFG, FakeAdapter)
    train, eval_ds = full_data.load_full_dataset(CFG, model_name="faketest")
    assert len(train) == 3  # 三个 train split 各存活 1 条,concatenate 后 3 条
    assert len(eval_ds) == 1
    assert set(train.column_names) >= {"input_features", "labels", "length"}


def test_load_full_dataset_guards(data_root):
    with pytest.raises(ValueError):
        full_data.load_full_dataset(CFG)  # 缺 model_name
    with pytest.raises(FileNotFoundError):
        full_data.load_full_dataset(CFG, model_name="nosuch")

    full_data.prepare_full_dataset(CFG, FakeAdapter)
    stale = copy.deepcopy(CFG)
    stale["model_size"] = "not-fake"
    with pytest.raises(ValueError):
        full_data.load_full_dataset(stale, model_name="faketest")


ADAPTER_CASES = [
    ("asrfs.whisper", "whisper"),
    ("asrfs.parakeet", "parakeet"),
]


@pytest.mark.parametrize("pkg_name,expected_model", ADAPTER_CASES)
def test_build_dataset_full_delegates_to_load_full_dataset(monkeypatch, pkg_name, expected_model):
    pkg = importlib.import_module(pkg_name)
    cfg = yaml.safe_load(Path(f"asrfs/{expected_model}/config.yaml").read_text())

    seen = {}

    def fake_load(cfg_in, model_name=None):
        seen["model_name"] = model_name
        return ("train-sentinel", "eval-sentinel")

    monkeypatch.setattr(full_data, "load_full_dataset", fake_load)
    out = pkg.build_dataset(cfg, None, mode="full")

    assert out == ("train-sentinel", "eval-sentinel")
    assert seen["model_name"] == expected_model
