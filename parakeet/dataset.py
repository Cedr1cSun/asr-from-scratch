"""Feature/label preparation, batch collation and CTC decoding for Parakeet."""

from dataclasses import dataclass

import torch
from transformers import ParakeetFeatureExtractor, ParakeetTokenizerFast


def prepare_example(
    sample: dict,
    feature_extractor: ParakeetFeatureExtractor,
    tokenizer: ParakeetTokenizerFast,
) -> dict:
    """Raw audio + transcript -> variable-length input_features + label ids.

    LibriSpeech transcripts are all-caps; the official tokenizer was fit on
    natural-case text, so lowercase first (scoring normalizes case anyway).
    No BOS/EOS — CTC labels are the bare token sequence.
    """
    features = feature_extractor(
        sample["audio_array"], sampling_rate=sample["sampling_rate"]
    ).input_features[0]
    labels = tokenizer(sample["text"].lower(), add_special_tokens=False)["input_ids"]
    return {"input_features": features, "labels": labels}


@dataclass
class ParakeetCollator:
    feature_extractor: ParakeetFeatureExtractor
    pad_label_id: int  # = CTC blank; ParakeetForCTC masks labels == pad_token_id, NOT -100

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


def ctc_greedy_decode(
    ids_batch: torch.Tensor, tokenizer: ParakeetTokenizerFast, blank_id: int
) -> list[str]:
    """argmax id sequences -> texts: collapse repeats, drop blanks, decode."""
    texts = []
    for ids in ids_batch:
        collapsed = torch.unique_consecutive(ids)
        kept = collapsed[collapsed != blank_id].tolist()
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts
