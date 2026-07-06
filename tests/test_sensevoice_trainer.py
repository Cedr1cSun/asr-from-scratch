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


def test_augment_cfg_disables_group_by_length(tmp_path, tiny_model):
    # cfg 带 augment 段(full 模式标志)+ overrides 注入长度分桶采样 → 必须被剥离,
    # 否则 LengthGroupedSampler 每 epoch 扫 length 列会顺带触发 train split 的
    # set_transform 增广再丢弃(见 asrfs/common/full_data.py load_full_dataset)。
    cfg = {**CFG, "run_name": "sv_full_test", "augment": {"spec_augment": {"time_axis": 0, "p": 0.9}}}
    trainer = build_trainer(
        cfg, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={
            "output_dir": str(tmp_path),
            "train_sampling_strategy": "group_by_length",
            "length_column_name": "length",
        },
    )
    assert getattr(trainer.args, "train_sampling_strategy", None) != "group_by_length"


def test_no_augment_cfg_keeps_group_by_length(tmp_path, tiny_model):
    # 反向:cfg 无 augment 段(非 full 模式)→ overrides 的长度分桶原样保留。
    trainer = build_trainer(
        {**CFG, "run_name": "sv_no_aug_test"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={
            "output_dir": str(tmp_path),
            "train_sampling_strategy": "group_by_length",
            "length_column_name": "length",
        },
    )
    assert trainer.args.train_sampling_strategy == "group_by_length"
