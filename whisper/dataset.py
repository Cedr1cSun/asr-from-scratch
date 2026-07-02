"""Feature/label preparation and batch collation for Whisper training."""

from dataclasses import dataclass

import torch
from transformers import WhisperProcessor


def prepare_example(sample: dict, processor: WhisperProcessor) -> dict:
    """Raw audio + transcript -> input_features (80x3000) + label token ids.

    LibriSpeech transcripts are all-caps; Whisper's BPE was fit on natural-case
    text, so lowercase before tokenizing. Scoring normalizes case anyway.
    """
    features = processor.feature_extractor(
        sample["audio_array"], sampling_rate=sample["sampling_rate"]
    ).input_features[0]
    labels = processor.tokenizer(sample["text"].lower()).input_ids
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
            labels_batch["attention_mask"].ne(1), -100
        )
        # tokenizer already prepends <|startoftranscript|>; the model's
        # shift_tokens_right adds it again as decoder_start, so drop it here
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        out["labels"] = labels
        return out
