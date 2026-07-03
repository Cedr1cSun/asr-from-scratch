"""sensevoice 契约面 + collator/decode/checkpoint 往返(镜像 test_parakeet_adapter)。"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from asrfs.sensevoice import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    build_collator,
    build_dataset,
    build_model,
    decode,
    load_checkpoint,
    make_example,
    save_checkpoint,
)
from asrfs.sensevoice.model import SenseVoiceConfig, SenseVoiceForCTC

TINY = dict(
    vocab_size=11, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=64, fsmn_kernel_size=11, sanm_shift=0, lfr_m=7, lfr_n=6,
    num_mel_bins=8, dropout=0.0, blank_id=10,
)


class FakeTokenizer:
    vocab_size = 10

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 10 for c in text[:5]]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i) for i in ids)


class FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate):
        t = max(1, len(audio) // 160)
        feats = np.zeros((t, 8), dtype=np.float32)

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
    return SenseVoiceForCTC(SenseVoiceConfig(**TINY)).eval()


def test_constants():
    assert LOSS_FAMILY == "ctc"
    assert LABEL_PAD_ID == 1024
    assert EXPECTED_FROZEN == set()


def test_config_yaml_harness_keys():
    cfg = yaml.safe_load(Path("asrfs/sensevoice/config.yaml").read_text())
    for key in ("learning_rate", "warmup_steps", "max_steps",
                "per_device_train_batch_size", "gradient_accumulation_steps",
                "logging_steps", "eval_steps", "seed"):
        assert key in cfg["training"], key
    for key in ("n_train", "n_eval"):
        assert key in cfg["data"], key
    for key in ("overfit1_steps", "overfit1_lr", "mini100_steps", "mini100_lr"):
        assert key in cfg["smoke"], key
    assert cfg["model"]["hidden_size"] == 384
    assert cfg["model"]["num_hidden_layers"] == 16


@pytest.mark.slow
def test_label_pad_id_matches_tokenizer():
    from asrfs.sensevoice.model import build_tokenizer

    assert LABEL_PAD_ID == build_tokenizer().vocab_size


def test_make_example_and_collator_fake_processor():
    proc = FakeBundle()
    ex = make_example(proc, np.zeros(1600, dtype=np.float32), 16000, "hello")
    assert set(ex) == {"input_features", "labels"}
    collator = build_collator({}, proc, model=None)
    ex2 = make_example(proc, np.zeros(3200, dtype=np.float32), 16000, "hi")
    batch = collator([ex, ex2])
    assert batch["labels"].shape[0] == 2
    assert (batch["labels"] == proc.tokenizer.vocab_size).any()  # pad = blank = vocab_size


def test_build_dataset_bad_mode():
    with pytest.raises(ValueError):
        build_dataset({}, FakeBundle(), mode="nope")


def test_decode_tolerates_labels(tiny_model):
    proc = FakeBundle()
    batch = {
        "input_features": torch.randn(2, 24, 8),
        "attention_mask": torch.ones(2, 24, dtype=torch.long),
        "labels": torch.full((2, 3), 10, dtype=torch.long),
    }
    texts = decode(tiny_model, proc, batch)
    assert isinstance(texts, list) and len(texts) == 2
    assert all(isinstance(t, str) for t in texts)


def test_checkpoint_roundtrip_model_only(tmp_path, tiny_model):
    out = tmp_path / "ckpt"
    tiny_model.save_pretrained(out)
    reloaded = SenseVoiceForCTC.from_pretrained(out).eval()
    feats = torch.randn(1, 24, 8)
    a = tiny_model(input_features=feats).logits
    b = reloaded(input_features=feats).logits
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.slow
def test_checkpoint_roundtrip_with_processor(tmp_path, tiny_model):
    from asrfs.sensevoice import build_processor

    proc = build_processor({})
    out = tmp_path / "ckpt"
    save_checkpoint(tiny_model, proc, str(out))
    model2, proc2 = load_checkpoint({}, str(out))
    assert proc2.tokenizer.vocab_size == proc.tokenizer.vocab_size
    feats = torch.randn(1, 24, 8)
    assert torch.allclose(
        tiny_model(input_features=feats).logits,
        model2(input_features=feats).logits,
        atol=1e-6,
    )
