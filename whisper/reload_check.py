"""Verify a save_pretrained checkpoint reloads and transcribes end-to-end.

This mirrors what a SURE-EVAL ModelWrapper would do: from_pretrained(dir),
predict(audio) -> text. Run: python -m whisper.reload_check outputs/<run>/final
"""

import sys

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common.data import fetch_smoke_subset


def main(ckpt_dir: str) -> None:
    device = torch.device("cuda")
    processor = WhisperProcessor.from_pretrained(ckpt_dir)  # local dir, not hub
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir).to(device)
    sample = fetch_smoke_subset(n=8)[0]
    features = processor.feature_extractor(
        sample["audio_array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(features, max_new_tokens=200)
    text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f"ref: {sample['text']}")
    print(f"hyp: {text}")
    print("RELOAD CHECK: OK (non-empty)" if text.strip() else "RELOAD CHECK: EMPTY OUTPUT")


if __name__ == "__main__":
    main(sys.argv[1])
