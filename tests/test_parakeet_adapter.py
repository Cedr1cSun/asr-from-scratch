from pathlib import Path

import numpy as np
import pytest
import yaml

import asrfs.parakeet.dataset as pds
from asrfs.common.data import fetch_smoke_subset
from asrfs.parakeet import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    ParakeetProcessorBundle,
    build_collator,
    build_dataset,
    build_model,
    build_processor,
    make_example,
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


def test_build_dataset_full_not_implemented():
    with pytest.raises(NotImplementedError):
        build_dataset({}, None, mode="full")


def test_build_dataset_bad_mode():
    with pytest.raises(ValueError):
        build_dataset({}, None, mode="overfit2")


@pytest.mark.slow
def test_make_example_and_build_collator(processor_bundle):
    ex1 = make_example(processor_bundle, np.zeros(16000 * 2, dtype=np.float32), 16000, "HELLO WORLD")
    ex2 = make_example(
        processor_bundle, np.zeros(16000 * 4, dtype=np.float32), 16000, "A LONGER FAKE SENTENCE"
    )
    assert set(ex1) == {"input_features", "labels"}
    collator = build_collator({}, processor_bundle, model=None)
    batch = collator([ex1, ex2])
    assert batch["input_features"].shape[0] == 2
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()
    blank = processor_bundle.tokenizer.vocab_size
    lens = [len(ex1["labels"]), len(ex2["labels"])]
    assert batch["labels"].shape[1] == max(lens)
    assert (batch["labels"][0, lens[0]:] == blank).all()


@pytest.mark.slow
def test_build_dataset_overfit1_and_mini100(processor_bundle):
    expected_cols = {"input_features", "labels", "id", "text", "length"}
    cfg = {"data": {"n_train": 100, "n_eval": 20}}
    train_ds, eval_ds = build_dataset(cfg, processor_bundle, mode="overfit1")
    assert eval_ds is None
    assert len(train_ds) == 100
    assert train_ds[0]["labels"] == train_ds[99]["labels"]  # 同一样本 ×100
    assert set(train_ds.column_names) == expected_cols
    row = train_ds[0]
    assert isinstance(row["id"], str) and row["id"]
    assert isinstance(row["text"], str) and row["text"]
    # length = 原始音频采样点数(override,A4 定案),非 len(input_features)
    overfit_raw = fetch_smoke_subset(n=8)[0]
    assert row["length"] == len(overfit_raw["audio_array"])
    train_ds, eval_ds = build_dataset(cfg, processor_bundle, mode="mini100")
    assert len(train_ds) == 100 and len(eval_ds) == 20
    assert set(train_ds.column_names) == expected_cols
    assert set(eval_ds.column_names) == expected_cols
    mini_raw = fetch_smoke_subset(n=120)[100]
    assert eval_ds[0]["length"] == len(mini_raw["audio_array"])


@pytest.mark.slow
def test_prepare_length_is_raw_audio_sample_count(processor_bundle):
    """length 必须是原始音频采样点数(A4 定案,供 group_by_length 按时长分桶),
    而非 len(input_features);两条不同时长样本的 length 必须不同且各自等于采样点数。"""
    short = {
        "id": "fake-0",
        "audio_array": np.zeros(1600, dtype=np.float32),
        "sampling_rate": 16000,
        "text": "short clip",
    }
    long_ = {
        "id": "fake-1",
        "audio_array": np.zeros(16000, dtype=np.float32),
        "sampling_rate": 16000,
        "text": "a much longer fake clip here",
    }
    fe, tok = processor_bundle.feature_extractor, processor_bundle.tokenizer
    row_short = pds._to_row(short, fe, tok)
    row_long = pds._to_row(long_, fe, tok)
    assert row_short["length"] == 1600
    assert row_long["length"] == 16000
    assert row_short["length"] != row_long["length"]
