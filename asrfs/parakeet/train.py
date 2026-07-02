import argparse
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

from asrfs.common.data import fetch_smoke_subset
from asrfs.common.metrics import wer
from asrfs.parakeet.dataset import ParakeetCollator, ctc_greedy_decode, prepare_example
from asrfs.parakeet.model import build_feature_extractor, build_model, build_tokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="asrfs/parakeet/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size

    fe, tok = build_feature_extractor(), build_tokenizer()
    model = build_model()
    blank = tok.vocab_size

    n_train, n_eval = cfg["data"]["n_train"], cfg["data"]["n_eval"]
    raw = fetch_smoke_subset(n=n_train + n_eval)
    prepared = raw.map(
        lambda s: prepare_example(s, fe, tok),
        remove_columns=raw.column_names,
    )
    train_ds = prepared.select(range(n_train))
    eval_ds = prepared.select(range(n_train, n_train + n_eval))

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(-1)

    def compute_metrics(pred):
        preds = torch.as_tensor(pred.predictions)
        preds[preds < 0] = blank
        hyps = ctc_greedy_decode(preds, tok, blank_id=blank)
        refs = []
        for ids in pred.label_ids:
            kept = [i for i in ids.tolist() if i != blank and i >= 0]
            refs.append(tok.decode(kept, skip_special_tokens=True))
        return {"wer": wer(refs, hyps)}

    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=f"outputs/{cfg['run_name']}",
        learning_rate=float(t_cfg["learning_rate"]),
        warmup_steps=t_cfg["warmup_steps"],
        max_steps=args.max_steps or t_cfg["max_steps"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ParakeetCollator(fe, pad_label_id=blank),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    out = Path(f"outputs/{cfg['run_name']}/final")
    model.save_pretrained(out)
    fe.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
