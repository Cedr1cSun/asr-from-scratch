"""x_asr 契约面 + collator 复用 + decode + checkpoint(镜像 sensevoice adapter 测试)。"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from asrfs.x_asr import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    build_collator,
    build_dataset,
    decode,
    make_example,
)
from asrfs.x_asr.model import XASRConfig, XASRForRNNT

TINY = dict(
    vocab_size=11, blank_id=10, num_mel_bins=80,
    downsampling_factor=(1, 2), num_encoder_layers=(1, 1),
    encoder_dim=(24, 32), encoder_unmasked_dim=(24, 24),
    query_head_dim=8, pos_head_dim=4, value_head_dim=8, pos_dim=16,
    num_heads=(2, 2), feedforward_dim=(48, 48), cnn_module_kernel=(7, 7),
    decoder_dim=16, joiner_dim=16, context_size=2, dropout=0.0,
)


class FakeTokenizer:
    vocab_size = 10

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 10 for c in text[:5]]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i) for i in ids)


class FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate):
        t = max(9, len(audio) // 160)
        feats = np.zeros((t, 80), dtype=np.float32)

        class R:
            input_features = [feats]

        return R()

    def pad(self, features, return_tensors="pt", return_attention_mask=True):
        arrs = [np.asarray(f["input_features"]) for f in features]
        t_max = max(a.shape[0] for a in arrs)
        batch = torch.zeros(len(arrs), t_max, arrs[0].shape[1])
        mask = torch.zeros(len(arrs), t_max, dtype=torch.long)
        for i, a in enumerate(arrs):
            batch[i, : a.shape[0]] = torch.from_numpy(a)
            mask[i, : a.shape[0]] = 1
        return {"input_features": batch, "attention_mask": mask}


class FakeBundle:
    tokenizer = FakeTokenizer()
    feature_extractor = FakeFeatureExtractor()


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return XASRForRNNT(XASRConfig(**TINY)).eval()


def test_constants():
    assert LOSS_FAMILY == "rnnt"
    assert LABEL_PAD_ID == 1024
    assert EXPECTED_FROZEN == set()


def test_config_yaml_harness_keys():
    cfg = yaml.safe_load(Path("asrfs/x_asr/config.yaml").read_text())
    for key in ("learning_rate", "warmup_steps", "max_steps",
                "per_device_train_batch_size", "gradient_accumulation_steps",
                "logging_steps", "eval_steps", "seed"):
        assert key in cfg["training"], key
    for key in ("n_train", "n_eval"):
        assert key in cfg["data"], key
    for key in ("overfit1_steps", "overfit1_lr", "mini100_steps", "mini100_lr"):
        assert key in cfg["smoke"], key
    assert cfg["model"] == {}  # 结构走 SMALL_MODEL 缺省;model 段留空自由区


@pytest.mark.slow
def test_label_pad_id_matches_tokenizer():
    from asrfs.x_asr.model import build_tokenizer

    assert LABEL_PAD_ID == build_tokenizer().vocab_size


def test_make_example_and_collator_reuse():
    proc = FakeBundle()
    ex = make_example(proc, np.zeros(3200, dtype=np.float32), 16000, "hello")
    assert set(ex) == {"input_features", "labels"}
    collator = build_collator({}, proc, model=None)
    from asrfs.common.ctc import CTCCollator

    assert isinstance(collator, CTCCollator)
    ex2 = make_example(proc, np.zeros(6400, dtype=np.float32), 16000, "hi")
    batch = collator([ex, ex2])
    assert (batch["labels"] == proc.tokenizer.vocab_size).any()


def test_build_dataset_bad_mode():
    with pytest.raises(ValueError):
        build_dataset({}, FakeBundle(), mode="nope")


def test_decode_returns_strings(tiny_model):
    proc = FakeBundle()
    batch = {
        "input_features": torch.randn(2, 41, 80),
        "attention_mask": torch.ones(2, 41, dtype=torch.long),
        "labels": torch.full((2, 3), 10, dtype=torch.long),
    }
    texts = decode(tiny_model, proc, batch)
    assert isinstance(texts, list) and len(texts) == 2
    assert all(isinstance(t, str) for t in texts)
