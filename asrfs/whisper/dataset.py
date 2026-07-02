from dataclasses import dataclass

from datasets import Dataset
from transformers import WhisperProcessor

from asrfs.common.data import fetch_smoke_subset
from asrfs.whisper.model import LABEL_PAD_ID


def make_example(processor, audio, sampling_rate: int, text: str) -> dict:
    """适配契约 make_example:单条 (音频, 文本) -> collator 可直接吃的 example。"""
    features = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    labels = processor.tokenizer(text.lower()).input_ids
    return {"input_features": features, "labels": labels}


@dataclass
class WhisperCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        feature_rows = [{"input_features": ex["input_features"]} for ex in batch]
        out = self.processor.feature_extractor.pad(feature_rows, return_tensors="pt")

        label_rows = [{"input_ids": ex["labels"]} for ex in batch]
        labels_batch = self.processor.tokenizer.pad(label_rows, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), LABEL_PAD_ID
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        out["labels"] = labels
        return out


def build_collator(cfg: dict, processor, model):
    """适配契约 build_collator:decoder_start_token_id 内部自取,消除调用方接线耦合。"""
    return WhisperCollator(processor, model.config.decoder_start_token_id)


def _prepare(raw: Dataset, processor) -> Dataset:
    def _to_row(s: dict) -> dict:
        ex = make_example(processor, s["audio_array"], s["sampling_rate"], s["text"])
        # 参考列(随行保留):id/text 供 harness run_smoke 写 overfit1 报告,
        # length 供 Trainer group_by_length;collator 只读已知键,多余列无害。
        ex["id"] = s["id"]
        ex["text"] = s["text"]
        ex["length"] = len(ex["input_features"])
        return ex

    return raw.map(_to_row, remove_columns=raw.column_names)


def build_dataset(cfg: dict, processor, mode: str) -> tuple:
    """适配契约 build_dataset:mode ∈ {overfit1, mini100, full};返回 (train_ds, eval_ds|None)。
    overfit1/mini100 每行含 input_features/labels + 参考列 id/text/length。"""
    n_train, n_eval = cfg["data"]["n_train"], cfg["data"]["n_eval"]
    if mode == "overfit1":
        row = fetch_smoke_subset(n=n_train + n_eval)[0]
        raw = Dataset.from_list([row] * 100)
        return _prepare(raw, processor), None
    if mode == "mini100":
        raw = fetch_smoke_subset(n=n_train + n_eval)
        prepared = _prepare(raw, processor)
        train_ds = prepared.select(range(n_train))
        eval_ds = prepared.select(range(n_train, n_train + n_eval))
        return train_ds, eval_ds
    if mode == "full":
        raise NotImplementedError(
            "mode='full' 依赖 960h 数据管线(asrfs.common 的 prepare_full_dataset/"
            "load_full_dataset),随 Task A9 落地;当前请用 'overfit1' 或 'mini100'。"
        )
    raise ValueError(f"unknown mode {mode!r}; expected one of 'overfit1', 'mini100', 'full'")
