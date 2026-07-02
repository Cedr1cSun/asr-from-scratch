import re

import numpy as np
import pytest
from datasets import Dataset

import asrfs.whisper.dataset as wds
from asrfs.common.data import LIBRISPEECH_REVISION
from asrfs.whisper.dataset import build_collator, build_dataset, make_example
from asrfs.whisper.model import LABEL_PAD_ID, build_processor

FAKE_TEXTS = ["hello world", "a longer fake sentence", "third sample"]


def _fake_sample(i: int, text: str, n_samples: int = 16000) -> dict:
    rng = np.random.default_rng(i)
    return {
        "id": f"fake-{i}",
        "audio_array": (0.01 * rng.standard_normal(n_samples)).astype(np.float32),
        "sampling_rate": 16000,
        "text": text,
    }


def _fake_fetch(n: int) -> Dataset:
    return Dataset.from_list([_fake_sample(i, FAKE_TEXTS[i % len(FAKE_TEXTS)]) for i in range(n)])


def _cfg() -> dict:
    return {"data": {"n_train": 2, "n_eval": 1}}


class _FakeModelConfig:
    decoder_start_token_id = 50257


class _FakeModel:
    config = _FakeModelConfig()


@pytest.fixture(scope="module")
def processor():
    return build_processor({"model": {"size": "medium"}})


def test_librispeech_revision_is_pinned_sha():
    assert LIBRISPEECH_REVISION != "main"
    assert re.fullmatch(r"[0-9a-f]{40}", LIBRISPEECH_REVISION), LIBRISPEECH_REVISION


def test_make_example_shapes_and_lowercased_labels(processor):
    s = _fake_sample(0, "HELLO WORLD")
    ex = make_example(processor, s["audio_array"], s["sampling_rate"], s["text"])
    assert np.asarray(ex["input_features"]).shape == (80, 3000)
    assert ex["labels"] == processor.tokenizer("hello world").input_ids


def test_collator_pads_and_strips_sot(processor):
    exs = [
        make_example(processor, _fake_sample(0, "hi")["audio_array"], 16000, "hi"),
        make_example(processor, _fake_sample(1, FAKE_TEXTS[1])["audio_array"], 16000, FAKE_TEXTS[1]),
    ]
    collator = build_collator({}, processor, _FakeModel())
    batch = collator(exs)
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    # SOT(= decoder_start_token_id 50257)已从 labels 首列剥离
    assert (batch["labels"][:, 0] != 50257).all()
    # 短句尾部按 LABEL_PAD_ID 填充(剥 SOT 后长度 = 原 labels 长度 - 1)
    short_len = len(exs[0]["labels"]) - 1
    assert (batch["labels"][0, short_len:] == LABEL_PAD_ID).all()


def test_build_dataset_overfit1(processor, monkeypatch):
    monkeypatch.setattr(wds, "fetch_smoke_subset", _fake_fetch)
    train_ds, eval_ds = build_dataset(_cfg(), processor, mode="overfit1")
    assert eval_ds is None
    assert len(train_ds) == 100
    assert train_ds[0]["labels"] == train_ds[99]["labels"]
    assert set(train_ds.column_names) == {"input_features", "labels", "id", "text", "length"}
    # 参考列:harness run_smoke 读 id/text 写 overfit1 报告
    assert train_ds[0]["id"] == "fake-0"
    assert train_ds[0]["text"] == FAKE_TEXTS[0]
    assert train_ds[99]["id"] == "fake-0"  # 同一条 ×100
    assert isinstance(train_ds[0]["length"], int)
    assert train_ds[0]["length"] == 16000  # length = 原始音频采样点数(1s@16kHz),非 mel-bin 轴长度 80


def test_build_dataset_mini100_split(processor, monkeypatch):
    monkeypatch.setattr(wds, "fetch_smoke_subset", _fake_fetch)
    train_ds, eval_ds = build_dataset(_cfg(), processor, mode="mini100")
    assert len(train_ds) == 2
    assert len(eval_ds) == 1
    assert set(train_ds.column_names) == {"input_features", "labels", "id", "text", "length"}
    assert set(eval_ds.column_names) == {"input_features", "labels", "id", "text", "length"}
    assert [row["id"] for row in train_ds] == ["fake-0", "fake-1"]
    assert eval_ds[0]["text"] == FAKE_TEXTS[2]
    assert eval_ds[0]["length"] == 16000  # length = 原始音频采样点数(1s@16kHz),非 mel-bin 轴长度 80


def test_prepare_length_is_raw_audio_sample_count(processor):
    """length 必须是原始音频采样点数,供 group_by_length 按时长分桶;
    WhisperFeatureExtractor 恒定输出 (80, 3000),若仍取 len(input_features) 则每行恒为 80,无分桶意义。"""
    short = _fake_sample(0, "short clip", n_samples=1600)
    long_ = _fake_sample(1, "a much longer fake clip here", n_samples=16000)
    raw = Dataset.from_list([short, long_])

    prepared = wds._prepare(raw, processor)

    assert prepared[0]["length"] == 1600
    assert prepared[1]["length"] == 16000
    assert prepared[0]["length"] != prepared[1]["length"]


def test_build_dataset_full_missing_manifest_raises(monkeypatch, tmp_path):
    """full 分支已委托 load_full_dataset:manifest 缺失 => FileNotFoundError(取代旧 NotImplementedError 断言)。"""
    import yaml

    import asrfs.whisper as pkg

    cfg = yaml.safe_load(open("asrfs/whisper/config.yaml"))
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        pkg.build_dataset(cfg, None, mode="full")


def test_build_dataset_unknown_mode(processor):
    with pytest.raises(ValueError):
        build_dataset(_cfg(), processor, mode="bogus")
