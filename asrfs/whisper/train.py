import argparse
from pathlib import Path

import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from asrfs.common.metrics import wer
from asrfs.whisper.dataset import build_collator, build_dataset
from asrfs.whisper.model import build_model, build_processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--max-steps", type=int, default=None, help="override config, for dry runs")
    parser.add_argument("--lr", type=float, default=None, help="override config peak learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="override config run_name")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    processor = build_processor(cfg)
    model = build_model(cfg)

    train_ds, eval_ds = build_dataset(cfg, processor, mode="mini100")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(refs, hyps)}

    t, m = cfg["training"], cfg["model"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"outputs/{cfg['run_name']}",
        learning_rate=float(t["learning_rate"]),
        warmup_steps=t["warmup_steps"],
        max_steps=args.max_steps or t["max_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=m["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        predict_with_generate=True,
        generation_max_length=m["generation_max_length"],
        save_strategy="no",
        seed=t["seed"],
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collator(cfg, processor, model),
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
