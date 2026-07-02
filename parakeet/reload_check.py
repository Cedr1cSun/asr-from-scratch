"""Verify a Parakeet-CTC save_pretrained checkpoint reloads and transcribes.

Run: python -m parakeet.reload_check outputs/<run>/final
"""

import sys

import numpy as np
import torch
from transformers import ParakeetFeatureExtractor, ParakeetForCTC, ParakeetTokenizerFast

from common.data import fetch_smoke_subset
from parakeet.dataset import ctc_greedy_decode


def main(ckpt_dir: str) -> None:
    device = torch.device("cuda")
    fe = ParakeetFeatureExtractor.from_pretrained(ckpt_dir)
    tok = ParakeetTokenizerFast.from_pretrained(ckpt_dir)
    model = ParakeetForCTC.from_pretrained(ckpt_dir).to(device)
    sample = fetch_smoke_subset(n=8)[0]
    audio = np.asarray(sample["audio_array"], dtype=np.float32)
    inputs = fe(audio, sampling_rate=sample["sampling_rate"], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    text = ctc_greedy_decode(logits.argmax(-1), tok, blank_id=model.config.pad_token_id)[0]
    print(f"ref: {sample['text']}")
    print(f"hyp: {text}")
    print("RELOAD CHECK: OK (non-empty)" if text.strip() else "RELOAD CHECK: EMPTY OUTPUT")


if __name__ == "__main__":
    main(sys.argv[1])
