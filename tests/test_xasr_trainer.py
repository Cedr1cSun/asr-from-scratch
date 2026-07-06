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


def test_models_disable_accepts_loss_kwargs(tmp_path, tiny_model):
    """F3 结构钉:forward(**kwargs)(VAR_KEYWORD)会让 transformers 5.12 Trainer
    判定 model_accepts_loss_kwargs=True,从而跳过 training_step 里 loss/grad_accum
    的归一化。两个模型都声明类属性 accepts_loss_kwargs=False,Trainer 在 __init__
    直接读它(trainer.py model_accepts_loss_kwargs 分支)。任一模型回退到 **kwargs
    推断都会让此断言失败。"""
    from transformers import Trainer, TrainingArguments

    from asrfs.sensevoice.model import SenseVoiceConfig, SenseVoiceForCTC

    torch.manual_seed(0)
    sv = SenseVoiceForCTC(SenseVoiceConfig(
        vocab_size=11, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, fsmn_kernel_size=11, sanm_shift=0, lfr_m=7, lfr_n=6,
        num_mel_bins=8, dropout=0.0, blank_id=10,
    ))
    for m in (tiny_model, sv):
        assert m.accepts_loss_kwargs is False
        trainer = Trainer(
            model=m, args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        )
        assert trainer.model_accepts_loss_kwargs is False


def test_grad_accum_normalizes_step_loss(tmp_path, tiny_model):
    """F3 行为验证:accepts_loss_kwargs=False → training_step 按
    current_gradient_accumulation_steps 归一化 loss(仅在 num_items_in_batch 非空时
    才是 bug 触发条件)。grad_accum=2 的 step loss 必须是 grad_accum=1 的一半
    (micro-batch 均值),而不是等于它——后者即 **kwargs 触发的 G× 梯度放大 bug。
    每步前固定随机种子以钉住 zipformer 训练态的随机层丢弃。"""
    trainer = build_trainer(
        {**CFG, "run_name": "xa_ga"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={"output_dir": str(tmp_path)},
    )
    model = trainer.model
    torch.manual_seed(0)
    feats = torch.randn(2, 41, 80)
    mask = torch.ones(2, 41)
    labels = torch.full((2, 4), TINY["blank_id"], dtype=torch.long)
    labels[0, :3] = torch.tensor([1, 2, 3])
    labels[1, :4] = torch.tensor([4, 5, 6, 7])
    inputs = {"input_features": feats, "attention_mask": mask, "labels": labels}

    def step(g):
        model.zero_grad()
        trainer.current_gradient_accumulation_steps = g
        torch.manual_seed(1234)
        return trainer.training_step(
            model, {k: v.clone() for k, v in inputs.items()}, num_items_in_batch=10
        )

    loss1 = step(1)
    loss2 = step(2)
    assert torch.allclose(loss2 * 2, loss1, rtol=1e-4)


@pytest.fixture()
def tiny_trainer_factory(tmp_path, tiny_model):
    """局部 helper(非全局夹具):按既有 build_trainer 测试的构造方式(tiny model +
    空数据集),接受额外的 training 段覆盖键,拼进 CFG["training"]。"""
    def _factory(extra: dict):
        cfg = {**CFG, "training": {**CFG["training"], **extra}, "run_name": "xa_eden_test"}
        return build_trainer(
            cfg, tiny_model, FakeBundle(),
            train_ds=None, eval_ds=None, collator=lambda b: b,
            overrides={"output_dir": str(tmp_path)},
        )
    return _factory


def test_eden_params_default_unchanged(tiny_trainer_factory):
    trainer = tiny_trainer_factory({})  # cfg training 段无 lr_batches/lr_epochs
    sched = trainer.lr_scheduler
    assert sched.lr_batches == 7500 and sched.lr_epochs == 3.5


def test_eden_params_from_cfg(tiny_trainer_factory):
    trainer = tiny_trainer_factory({"lr_batches": 1234, "lr_epochs": 1.5})
    sched = trainer.lr_scheduler
    assert sched.lr_batches == 1234 and sched.lr_epochs == 1.5


def test_augment_cfg_disables_group_by_length(tmp_path, tiny_model):
    # cfg 带 augment 段(full 模式标志)+ overrides 注入长度分桶采样 → 必须被剥离
    # (见 asrfs/common/full_data.py load_full_dataset 的 augment-and-discard 问题)。
    cfg = {**CFG, "run_name": "xa_full_test", "augment": {"spec_augment": {"time_axis": 0, "p": 0.9}}}
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
    # 反向:cfg 无 augment 段 → overrides 的长度分桶原样保留。
    trainer = build_trainer(
        {**CFG, "run_name": "xa_no_aug_test"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={
            "output_dir": str(tmp_path),
            "train_sampling_strategy": "group_by_length",
            "length_column_name": "length",
        },
    )
    assert trainer.args.train_sampling_strategy == "group_by_length"


def test_on_epoch_end_advances_eden_epoch(tmp_path, tiny_model):
    """F6:Eden 的 epoch 衰减靠每 epoch 末 step_epoch() 驱动;HF Trainer 只调
    step()(→step_batch),epoch 因子会永远停在 0。BatchCountCallback.on_epoch_end
    从 CallbackHandler 传入的 lr_scheduler 上调 step_epoch。这里走真实
    callback_handler.on_epoch_end 接线(call_event 会把 lr_scheduler 传给回调),
    验证 Eden.epoch 被推进。"""
    trainer = build_trainer(
        {**CFG, "run_name": "xa_epoch"}, tiny_model, FakeBundle(),
        train_ds=None, eval_ds=None, collator=lambda b: b,
        overrides={"output_dir": str(tmp_path)},
    )
    assert isinstance(trainer.callback_handler.lr_scheduler, EdenForTrainer)
    sched = trainer.lr_scheduler
    start = sched.epoch
    trainer.callback_handler.on_epoch_end(trainer.args, trainer.state, trainer.control)
    assert sched.epoch == start + 1
    trainer.callback_handler.on_epoch_end(trainer.args, trainer.state, trainer.control)
    assert sched.epoch == start + 2
