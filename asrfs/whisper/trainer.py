import dataclasses

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from asrfs.common.metrics import wer
from asrfs.whisper.model import LABEL_PAD_ID

_LEGAL_FIELDS = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}


def _build_training_args(cfg: dict, overrides: dict, has_eval: bool) -> Seq2SeqTrainingArguments:
    """钉死的 overrides 语义:{**cfg_args, **overrides} 构造前合并;
    未知 TrainingArguments 字段抛 ValueError;remove_unused_columns=False 恒置。"""
    t, m = cfg["training"], cfg["model"]
    cfg_args = {
        "output_dir": f"outputs/{cfg['run_name']}",
        "learning_rate": float(t["learning_rate"]),
        "warmup_steps": t["warmup_steps"],
        "max_steps": t["max_steps"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "gradient_checkpointing": m["gradient_checkpointing"],
        "logging_steps": t["logging_steps"],
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": t["eval_steps"],
        "predict_with_generate": t.get("predict_with_generate", True),
        "generation_max_length": m["generation_max_length"],
        "save_strategy": "no",
        "seed": t.get("seed", 42),
        "report_to": ["tensorboard"],
    }
    merged = {**cfg_args, **overrides}
    unknown = sorted(set(merged) - _LEGAL_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown Seq2SeqTrainingArguments field(s): {unknown}; "
            f"overrides must be legal TrainingArguments fields"
        )
    merged["remove_unused_columns"] = False
    return Seq2SeqTrainingArguments(**merged)


def build_trainer(cfg: dict, model, processor, train_ds, eval_ds, collator,
                  overrides: dict) -> Seq2SeqTrainer:
    """适配契约 build_trainer:Seq2SeqTrainer + Seq2SeqTrainingArguments。"""
    args = _build_training_args(cfg, overrides, has_eval=eval_ds is not None)

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == LABEL_PAD_ID] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(refs, hyps)}

    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics
        if (eval_ds is not None and args.predict_with_generate)
        else None,
    )
