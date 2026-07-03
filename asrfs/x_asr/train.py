r"""x_asr 训练面:ScaledAdam + Eden(vendored)显式接进 HF Trainer。

recipe 事实(icefall train.py):ScaledAdam(get_parameter_groups_with_lrs(model,
lr=base_lr, include_names=True), lr=base_lr, clipping_scale=2.0);
Eden(opt, lr_batches=7500, lr_epochs=3.5, warmup_start=0.1);base_lr=0.045。
ScheduledFloat/内部 warmup 靠 set_batch_count(model, step) 驱动 -> 回调。

vendored optim.py 实测(grep -n "def step\|def get_last_lr" _vendor/optim.py):
LRScheduler 基类已有 state_dict/load_state_dict(纯 epoch/batch dict,直接透传即
兼容 HF Trainer 的 checkpoint 读写)和 get_lr/step_batch/step_epoch,但没有
`.step()`(torch 优化器风格)——HF Trainer 只调 `.step()`,所以补一个映射到
`step_batch()`。基类 `get_last_lr()` 读 `self._last_lr`,这个属性要等第一次
`_set_lrs()`(由 step_batch/step_epoch 触发)才存在,构造后立即调用会
AttributeError;因此覆写为直接读 `optimizer.param_groups`,构造后即可用,且
与 Eden 更新后的值天然同步。
"""

import argparse
import dataclasses
from pathlib import Path

import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments

from asrfs.x_asr._vendor.icefall_compat import (
    get_parameter_groups_with_lrs,
    set_batch_count,
)
from asrfs.x_asr._vendor.optim import Eden, ScaledAdam
from asrfs.x_asr.dataset import build_collator, build_dataset
from asrfs.x_asr.model import build_model, build_processor, save_checkpoint


class EdenForTrainer(Eden):
    """HF Trainer 只会调 .step()/.get_last_lr();映射到 Eden 的 step_batch 语义。"""

    def step(self, epoch=None):
        self.step_batch()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class BatchCountCallback(TrainerCallback):
    """icefall set_batch_count 等价物:每步开始把 global_step 写进各模块。"""

    def __init__(self, model):
        self._model = model

    def on_step_begin(self, args, state, control, **kwargs):
        set_batch_count(self._model, float(state.global_step))


def build_trainer(cfg, model, processor, train_ds, eval_ds, collator, overrides: dict):
    t_cfg = cfg["training"]
    base_lr = float(t_cfg["learning_rate"])

    cfg_args = {
        "output_dir": f"outputs/{cfg.get('run_name', 'x_asr')}",
        "learning_rate": base_lr,  # 展示用;实际 lr 由 ScaledAdam/Eden 决定
        "warmup_steps": t_cfg["warmup_steps"],
        "max_steps": t_cfg["max_steps"],
        "per_device_train_batch_size": t_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": t_cfg["gradient_accumulation_steps"],
        "logging_steps": t_cfg["logging_steps"],
        "seed": t_cfg.get("seed", 42),
        "save_strategy": "no",
        "report_to": ["tensorboard"],
    }
    if eval_ds is not None:
        cfg_args["eval_strategy"] = "steps"
        cfg_args["eval_steps"] = t_cfg["eval_steps"]
    merged = {**cfg_args, **overrides}
    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(f"unknown TrainingArguments keys: {sorted(unknown)}")
    merged["remove_unused_columns"] = False
    training_args = TrainingArguments(**merged)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=base_lr, include_names=True),
        lr=base_lr,
        clipping_scale=2.0,
    )
    scheduler = EdenForTrainer(optimizer, lr_batches=7500, lr_epochs=3.5, warmup_start=0.1)
    # HF Trainer 在 optimizer.step() 之后才调 scheduler.step(),Eden 构造时不写 lr,
    # 首步会以未 warmup 的 base_lr 跑;先按 batch=0 写入 warmup 起点(icefall 语义)。
    scheduler.step_batch(0)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
        callbacks=[BatchCountCallback(model)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="asrfs/x_asr/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size

    processor = build_processor(cfg)
    model = build_model(cfg)
    train_ds, eval_ds = build_dataset(cfg, processor, mode="mini100")
    collator = build_collator(cfg, processor, model)
    trainer = build_trainer(cfg, model, processor, train_ds, eval_ds, collator, overrides)
    trainer.train()

    out = Path(f"outputs/{cfg['run_name']}/final")
    save_checkpoint(model, processor, str(out))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
