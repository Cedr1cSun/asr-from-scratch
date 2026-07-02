import argparse
import dataclasses
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

from asrfs.common.metrics import wer
from asrfs.parakeet.dataset import build_collator, build_dataset, ctc_greedy_decode
from asrfs.parakeet.model import build_model, build_processor, save_checkpoint


def build_trainer(cfg, model, processor, train_ds, eval_ds, collator, overrides: dict) -> Trainer:
    t_cfg = cfg["training"]
    tokenizer = processor.tokenizer
    blank = tokenizer.vocab_size

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(-1)

    def compute_metrics(pred):
        preds = torch.as_tensor(pred.predictions)
        preds[preds < 0] = blank
        hyps = ctc_greedy_decode(preds, tokenizer, blank_id=blank)
        refs = []
        for ids in pred.label_ids:
            kept = [i for i in ids.tolist() if i != blank and i >= 0]
            refs.append(tokenizer.decode(kept, skip_special_tokens=True))
        return {"wer": wer(refs, hyps)}

    cfg_args = {
        "output_dir": f"outputs/{cfg.get('run_name', 'parakeet')}",
        "learning_rate": float(t_cfg["learning_rate"]),
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
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="asrfs/parakeet/config.yaml")
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
        overrides["learning_rate"] = args.lr
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
