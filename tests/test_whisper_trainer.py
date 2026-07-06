import dataclasses

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from asrfs.whisper.dataset import build_collator, make_example
from asrfs.whisper.model import (
    build_model,
    build_processor,
    decode,
    load_checkpoint,
    save_checkpoint,
)
from asrfs.whisper.trainer import _build_training_args, build_trainer

HARNESS_OVERRIDE_KEYS = {
    "max_steps", "learning_rate", "lr_scheduler_type", "warmup_steps", "output_dir",
    "per_device_train_batch_size", "gradient_accumulation_steps", "logging_steps",
    "report_to", "seed", "save_strategy", "save_steps", "save_total_limit",
}


def _cfg() -> dict:
    return {
        "run_name": "unit_test",
        "model": {
            "size": "medium",
            "apply_spec_augment": False,
            "gradient_checkpointing": False,
            "generation_max_length": 2,
        },
        "training": {
            "learning_rate": 1.0e-4,
            "warmup_steps": 0,
            "max_steps": 10,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "eval_steps": 5,
            "seed": 42,
        },
        "data": {"n_train": 2, "n_eval": 1},
        "smoke": {"overfit1_steps": 300, "overfit1_lr": 3.0e-4,
                  "mini100_steps": 250, "mini100_lr": 1.0e-4},
    }


# ── overrides 合并语义(CPU-fast,不建模型)─────────────────────────────

def test_overrides_unknown_key_raises():
    with pytest.raises(ValueError, match="bogus_key"):
        _build_training_args(_cfg(), {"bogus_key": 1}, has_eval=False)


def test_overrides_take_precedence_and_pins(tmp_path):
    args = _build_training_args(
        _cfg(),
        {"max_steps": 7, "learning_rate": 5.0e-5, "output_dir": str(tmp_path), "report_to": []},
        has_eval=False,
    )
    assert args.max_steps == 7
    assert args.learning_rate == 5.0e-5
    assert args.output_dir == str(tmp_path)
    assert args.remove_unused_columns is False
    assert args.seed == 42
    assert args.eval_strategy == "no"


def test_harness_override_universe_is_legal():
    fields = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    assert HARNESS_OVERRIDE_KEYS <= fields


def test_predict_with_generate_default_true():
    from asrfs.whisper.trainer import _build_training_args
    args = _build_training_args(_cfg(), {}, has_eval=True)
    assert args.predict_with_generate is True


def test_predict_with_generate_off_disables_wer_metrics():
    from asrfs.whisper.trainer import _build_training_args
    base = _cfg()
    cfg = {**base, "training": {**base["training"], "predict_with_generate": False}}
    args = _build_training_args(cfg, {}, has_eval=True)
    assert args.predict_with_generate is False


def test_augment_cfg_disables_group_by_length():
    # cfg 带 augment 段(full 模式标志)+ overrides 注入长度分桶采样 → 必须被剥离
    # (见 asrfs/common/full_data.py load_full_dataset 的 augment-and-discard 问题)。
    cfg = {**_cfg(), "augment": {"spec_augment": {"time_axis": 1, "p": 0.9}}}
    args = _build_training_args(
        cfg,
        {"train_sampling_strategy": "group_by_length", "length_column_name": "length"},
        has_eval=False,
    )
    assert getattr(args, "train_sampling_strategy", None) != "group_by_length"


def test_no_augment_cfg_keeps_group_by_length():
    # 反向:cfg 无 augment 段 → overrides 的长度分桶原样保留。
    args = _build_training_args(
        _cfg(),
        {"train_sampling_strategy": "group_by_length", "length_column_name": "length"},
        has_eval=False,
    )
    assert args.train_sampling_strategy == "group_by_length"


# ── checkpoint 往返(tiny 配置,CPU-fast)───────────────────────────────

def test_checkpoint_roundtrip_tiny(tmp_path):
    tiny_cfg = WhisperConfig(
        vocab_size=64, num_mel_bins=80, max_source_positions=1500, max_target_positions=32,
        d_model=32, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=64,
        decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=2, decoder_start_token_id=1,
    )
    model = WhisperForConditionalGeneration(tiny_cfg)
    processor = build_processor({"model": {"size": "medium"}})
    ckpt = tmp_path / "ckpt"
    save_checkpoint(model, processor, str(ckpt))
    model2, processor2 = load_checkpoint({}, str(ckpt))
    params2 = dict(model2.named_parameters())
    for name, p in model.named_parameters():
        assert torch.allclose(p, params2[name], atol=1e-6), name
    assert processor2.tokenizer.pad_token_id == processor.tokenizer.pad_token_id


# ── 真模型路径(slow)────────────────────────────────────────────────

@pytest.fixture(scope="module")
def medium_setup():
    cfg = _cfg()
    processor = build_processor(cfg)
    model = build_model(cfg)
    return cfg, processor, model


def _tiny_ds(processor) -> Dataset:
    rng = np.random.default_rng(0)
    rows = []
    for text in ["hello world", "a second sample"]:
        audio = (0.01 * rng.standard_normal(16000)).astype(np.float32)
        rows.append(make_example(processor, audio, 16000, text))
    return Dataset.from_list(rows)


@pytest.mark.slow
def test_build_trainer_constructs(medium_setup, tmp_path):
    cfg, processor, model = medium_setup
    ds = _tiny_ds(processor)
    collator = build_collator(cfg, processor, model)
    trainer = build_trainer(
        cfg, model, processor, ds, None, collator,
        overrides={"output_dir": str(tmp_path), "report_to": []},
    )
    assert isinstance(trainer, Seq2SeqTrainer)
    assert trainer.args.remove_unused_columns is False
    assert trainer.args.max_steps == 10


@pytest.mark.slow
def test_predict_with_generate_false_disables_compute_metrics(medium_setup, tmp_path):
    cfg, processor, model = medium_setup
    ds = _tiny_ds(processor)
    collator = build_collator(cfg, processor, model)
    cfg_off = {**cfg, "training": {**cfg["training"], "predict_with_generate": False}}
    trainer = build_trainer(
        cfg_off, model, processor, ds, ds, collator,
        overrides={"output_dir": str(tmp_path), "report_to": []},
    )
    assert trainer.args.predict_with_generate is False
    assert trainer.compute_metrics is None


@pytest.mark.slow
def test_predict_with_generate_true_keeps_compute_metrics(medium_setup, tmp_path):
    cfg, processor, model = medium_setup
    ds = _tiny_ds(processor)
    collator = build_collator(cfg, processor, model)
    trainer = build_trainer(
        cfg, model, processor, ds, ds, collator,
        overrides={"output_dir": str(tmp_path), "report_to": []},
    )
    assert trainer.args.predict_with_generate is True
    assert trainer.compute_metrics is not None


@pytest.mark.slow
def test_decode_ignores_labels_and_returns_strs(medium_setup):
    cfg, processor, model = medium_setup
    ds = _tiny_ds(processor)
    collator = build_collator(cfg, processor, model)
    batch = collator([ds[0]])
    assert "labels" in batch  # decode 必须容忍并忽略
    out = decode(model, processor, batch)  # generation_max_length=2,CPU 可承受
    assert isinstance(out, list)
    assert len(out) == 1
    assert all(isinstance(t, str) for t in out)
