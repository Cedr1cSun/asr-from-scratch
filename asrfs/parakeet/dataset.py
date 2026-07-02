from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import ParakeetFeatureExtractor, ParakeetTokenizerFast

from asrfs.common.data import fetch_smoke_subset

OVERFIT1_REPLICAS = 100


def prepare_example(
    sample: dict,
    feature_extractor: ParakeetFeatureExtractor,
    tokenizer: ParakeetTokenizerFast,
) -> dict:
    audio = np.asarray(sample["audio_array"], dtype=np.float32)
    features = feature_extractor(audio, sampling_rate=sample["sampling_rate"]).input_features[0]
    labels = tokenizer(sample["text"].lower(), add_special_tokens=False)["input_ids"]
    return {"input_features": features, "labels": labels}


def make_example(processor, audio, sampling_rate: int, text: str) -> dict:
    sample = {"audio_array": audio, "sampling_rate": sampling_rate, "text": text}
    return prepare_example(sample, processor.feature_extractor, processor.tokenizer)


@dataclass
class ParakeetCollator:
    feature_extractor: ParakeetFeatureExtractor
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


def build_collator(cfg: dict, processor, model) -> ParakeetCollator:
    # blank(= CTC pad)从 processor 自取,调用方无需接线;model 参数契约占位
    # (whisper 侧要读 model.config.decoder_start_token_id,parakeet 用不上)。
    return ParakeetCollator(
        processor.feature_extractor, pad_label_id=processor.tokenizer.vocab_size
    )


def _to_row(sample: dict, fe, tok) -> dict:
    # 额外列契约(harness,R1):id/text 供 run_smoke 报告引用参考转写,
    # length: 原始音频采样点数(A4 定案,提取前算,与 whisper 一致的分桶语义),
    # 供 Trainer group_by_length(length_column_name="length")分桶;
    # collator 只读 input_features/labels,额外列无害。
    row = prepare_example(sample, fe, tok)
    row["id"] = str(sample["id"])
    row["text"] = sample["text"]
    row["length"] = len(sample["audio_array"])
    return row


def build_dataset(cfg: dict, processor, mode: str) -> tuple:
    if mode == "full":
        raise NotImplementedError(
            "mode='full' depends on asrfs.common.prepare_full_dataset (Task A9)"
        )
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


def ctc_greedy_decode(
    ids_batch: torch.Tensor, tokenizer: ParakeetTokenizerFast, blank_id: int
) -> list[str]:
    texts = []
    for ids in ids_batch:
        collapsed = torch.unique_consecutive(ids)
        kept = collapsed[collapsed != blank_id].tolist()
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts


def decode(model, processor, batch) -> list[str]:
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
