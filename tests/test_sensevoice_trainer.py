"""sensevoice build_trainer(共享 build_ctc_trainer 的接线)。"""

import pytest
import torch

from asrfs.sensevoice.model import SenseVoiceConfig, SenseVoiceForCTC
from asrfs.sensevoice.train import build_trainer

TINY = dict(
    vocab_size=11, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=64, fsmn_kernel_size=11, sanm_shift=0, lfr_m=7, lfr_n=6,
    num_mel_bins=8, dropout=0.0, blank_id=10,
)

CFG = {
    "training": {
        "learning_rate": 1e-4, "warmup_steps": 1, "max_steps": 2,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
        "logging_steps": 1, "eval_steps": 1, "seed": 42,
    },
}


class FakeTokenizer:
    vocab_size = 10

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class FakeBundle:
    tokenizer = FakeTokenizer()
    feature_extractor = None


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return SenseVoiceForCTC(SenseVoiceConfig(**TINY))


def test_build_trainer_defaults(tmp_path, tiny_model):
    trainer = build_trainer(
        {**CFG, "run_name": "sv_test"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={"output_dir": str(tmp_path)},
    )
    assert trainer.args.max_steps == 2
    assert trainer.args.remove_unused_columns is False
    assert trainer.compute_metrics is None  # eval_ds=None 时不接 metrics


def test_build_trainer_unknown_override_raises(tmp_path, tiny_model):
    with pytest.raises(ValueError, match="unknown TrainingArguments"):
        build_trainer(
            CFG, tiny_model, FakeBundle(), None, None, lambda b: b,
            overrides={"definitely_not_a_field": 1},
        )
