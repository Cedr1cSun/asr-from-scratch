"""Seq2SeqTrainer entry point.

Run from project root:  python -m whisper.train --config whisper/config.yaml
"""

import argparse
from pathlib import Path

import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from common.data import fetch_smoke_subset
from common.metrics import wer
from whisper.dataset import WhisperCollator, prepare_example
from whisper.model import build_model, build_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="whisper/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="override config, for dry runs")
    parser.add_argument("--lr", type=float, default=None, help="override config peak learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="override config run_name")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    processor = build_processor(cfg["model_size"])
    model = build_model(cfg["model_size"], apply_spec_augment=cfg["apply_spec_augment"])

    n_train, n_eval = cfg["data"]["n_train"], cfg["data"]["n_eval"]
    raw = fetch_smoke_subset(n=n_train + n_eval)
    prepared = raw.map(
        lambda s: prepare_example(s, processor),
        remove_columns=raw.column_names,
    )
    train_ds = prepared.select(range(n_train))
    eval_ds = prepared.select(range(n_train, n_train + n_eval))

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(refs, hyps)}

    t = cfg["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"outputs/{cfg['run_name']}",
        learning_rate=float(t["learning_rate"]),
        warmup_steps=t["warmup_steps"],
        max_steps=args.max_steps or t["max_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        predict_with_generate=True,
        generation_max_length=t["generation_max_length"],
        save_strategy="no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    out = Path(f"outputs/{cfg['run_name']}/final")
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
