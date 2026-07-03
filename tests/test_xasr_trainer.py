"""x_asr build_trainer:ScaledAdam+Eden 接线、overrides 校验、batch_count 回调。"""

import pytest
import torch

from asrfs.x_asr.model import XASRConfig, XASRForRNNT
from asrfs.x_asr.train import BatchCountCallback, EdenForTrainer, build_trainer

TINY = dict(
    vocab_size=11, blank_id=10, num_mel_bins=80,
    downsampling_factor=(1, 2), num_encoder_layers=(1, 1),
    encoder_dim=(24, 32), encoder_unmasked_dim=(24, 24),
    query_head_dim=8, pos_head_dim=4, value_head_dim=8, pos_dim=16,
    num_heads=(2, 2), feedforward_dim=(48, 48), cnn_module_kernel=(7, 7),
    decoder_dim=16, joiner_dim=16, context_size=2, dropout=0.0,
)

CFG = {
    "training": {
        "learning_rate": 0.045, "warmup_steps": 1, "max_steps": 2,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
        "logging_steps": 1, "eval_steps": 1, "seed": 42,
    },
}


class FakeBundle:
    tokenizer = None
    feature_extractor = None


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return XASRForRNNT(XASRConfig(**TINY))


def test_build_trainer_wires_scaled_adam_and_eden(tmp_path, tiny_model):
    trainer = build_trainer(
        {**CFG, "run_name": "xa_test"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={"output_dir": str(tmp_path)},
    )
    from asrfs.x_asr._vendor.optim import ScaledAdam

    assert isinstance(trainer.optimizer, ScaledAdam)
    assert isinstance(trainer.lr_scheduler, EdenForTrainer)
    assert trainer.args.remove_unused_columns is False
    assert any(isinstance(cb, BatchCountCallback) for cb in trainer.callback_handler.callbacks)


def test_eden_adapter_step_and_last_lr(tiny_model):
    from asrfs.x_asr._vendor.icefall_compat import get_parameter_groups_with_lrs
    from asrfs.x_asr._vendor.optim import ScaledAdam

    opt = ScaledAdam(
        get_parameter_groups_with_lrs(tiny_model, lr=0.045, include_names=True),
        lr=0.045, clipping_scale=2.0,
    )
    sched = EdenForTrainer(opt, lr_batches=7500, lr_epochs=3.5, warmup_start=0.1)
    lr0 = sched.get_last_lr()
    sched.step()
    lr1 = sched.get_last_lr()
    assert isinstance(lr0, list) and isinstance(lr1, list) and len(lr0) >= 1
    assert lr1[0] > 0


def test_build_trainer_unknown_override_raises(tmp_path, tiny_model):
    with pytest.raises(ValueError, match="unknown TrainingArguments"):
        build_trainer(
            CFG, tiny_model, FakeBundle(), None, None, lambda b: b,
            overrides={"definitely_not_a_field": 1},
        )
