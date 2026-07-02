from dataclasses import dataclass

import numpy as np
import torch
from transformers import ParakeetFeatureExtractor, ParakeetTokenizerFast

def prepare_example(
    sample: dict,
    feature_extractor: ParakeetFeatureExtractor,
    tokenizer: ParakeetTokenizerFast,
) -> dict:
    audio = np.asarray(sample["audio_array"], dtype=np.float32)
    features = feature_extractor(audio, sampling_rate=sample["sampling_rate"]).input_features[0]
    labels = tokenizer(sample["text"].lower(), add_special_tokens=False)["input_ids"]
    return {"input_features": features, "labels": labels}

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

def ctc_greedy_decode(
    ids_batch: torch.Tensor, tokenizer: ParakeetTokenizerFast, blank_id: int
) -> list[str]:
    texts = []
    for ids in ids_batch:
        collapsed = torch.unique_consecutive(ids)
        kept = collapsed[collapsed != blank_id].tolist()
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts
