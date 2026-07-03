"""CTC 族(parakeet / sensevoice)共享件:同 tokenizer/特征源,collator、解码、
smoke 数据集、Trainer 构造完全同构,按用户执行策略"按 loss 族写一套"抽取于此。

full 模式的 load_full_dataset 保持函数内局部 import(与抽取前语义一致,
测试对 asrfs.common.full_data 的 monkeypatch 仍然生效)。
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from asrfs.common.data import fetch_smoke_subset
from asrfs.common.metrics import wer

OVERFIT1_REPLICAS = 100


@dataclass
class CTCProcessorBundle:
    tokenizer: Any
    feature_extractor: Any


def prepare_ctc_example(sample: dict, feature_extractor, tokenizer) -> dict:
    audio = np.asarray(sample["audio_array"], dtype=np.float32)
    features = feature_extractor(audio, sampling_rate=sample["sampling_rate"]).input_features[0]
    labels = tokenizer(sample["text"].lower(), add_special_tokens=False)["input_ids"]
    return {"input_features": features, "labels": labels}


@dataclass
class CTCCollator:
    feature_extractor: Any
    pad_label_id: int

    def __call__(self, batch: list[dict]) -> dict:
        out = self.feature_extractor.pad(
            [{"input_features": ex["input_features"]} for ex in batch],
            return_tensors="pt",
            return_attention_mask=True,
        )
        max_len = max(len(ex["labels"]) for ex in batch)
        labels = torch.full((len(batch), max_len), self.pad_label_id, dtype=torch.long)
        for i, ex in enumerate(batch):
            labels[i, : len(ex["labels"])] = torch.as_tensor(ex["labels"], dtype=torch.long)
        out["labels"] = labels
        return out


def _to_row(sample: dict, fe, tok) -> dict:
    # 额外列契约(harness,R1):id/text 供 run_smoke 报告引用参考转写,
    # length: 原始音频采样点数(A4 定案,提取前算,与 whisper 一致的分桶语义),
    # 供 Trainer group_by_length(length_column_name="length")分桶;
    # collator 只读 input_features/labels,额外列无害。
    row = prepare_ctc_example(sample, fe, tok)
    row["id"] = str(sample["id"])
    row["text"] = sample["text"]
    row["length"] = len(sample["audio_array"])
    return row


def build_ctc_dataset(cfg: dict, processor, mode: str, model_name: str) -> tuple:
    if mode == "full":
        from asrfs.common.full_data import load_full_dataset

        return load_full_dataset(cfg, model_name=model_name)
    if mode not in ("overfit1", "mini100"):
        raise ValueError(f"unknown mode: {mode!r}")
    fe, tok = processor.feature_extractor, processor.tokenizer
    if mode == "overfit1":
        # 钉死 train.clean.100 首条(index 0);n=8 复用 golden smoke 既有缓存目录。
        sample = fetch_smoke_subset(n=8)[0]
        row = _to_row(sample, fe, tok)
        return Dataset.from_list([row] * OVERFIT1_REPLICAS), None
    n_train, n_eval = cfg["data"]["n_train"], cfg["data"]["n_eval"]
    raw = fetch_smoke_subset(n=n_train + n_eval)
    prepared = raw.map(
        lambda s: _to_row(s, fe, tok),
        remove_columns=raw.column_names,
    )
    return (
        prepared.select(range(n_train)),
        prepared.select(range(n_train, n_train + n_eval)),
    )


def ctc_greedy_decode(ids_batch: torch.Tensor, tokenizer, blank_id: int) -> list[str]:
    texts = []
    for ids in ids_batch:
        collapsed = torch.unique_consecutive(ids)
        kept = collapsed[collapsed != blank_id].tolist()
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts


def ctc_decode(model, processor, batch) -> list[str]:
    """契约解码入口:forward → argmax → CTC 折叠 → detokenize。

    batch = collator 产物(已在模型设备),可能含 labels 等训练键,一律忽略。
    """
    device = next(model.parameters()).device
    inputs = {"input_features": batch["input_features"].to(device)}
    if "attention_mask" in batch:
        inputs["attention_mask"] = batch["attention_mask"].to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(**inputs).logits
    finally:
        if was_training:
            model.train()
    blank = processor.tokenizer.vocab_size
    return ctc_greedy_decode(logits.argmax(-1), processor.tokenizer, blank_id=blank)


def build_ctc_trainer(
    cfg, model, processor, train_ds, eval_ds, collator, overrides: dict, default_run_name: str
) -> Trainer:
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
        "output_dir": f"outputs/{cfg.get('run_name', default_run_name)}",
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
