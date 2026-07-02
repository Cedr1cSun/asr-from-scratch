"""Smoke round 1: overfit a single utterance in FP32.

Random-init whisper-small must drive loss to ~0 on one sample and greedy-decode
the exact transcript back. Passing proves model/label/collator wiring; failing
means a plumbing bug (label shift, padding, masking), not a modeling problem.

Run from project root:  python -m whisper.smoke
"""

import argparse
import re
import time

import torch

from common.data import fetch_smoke_subset
from whisper.dataset import WhisperCollator, prepare_example
from whisper.model import build_model, build_processor, init_report

PUNCT = re.compile(r"[^A-Z' ]")


def rough_normalize(text: str) -> str:
    """Placeholder for common/metrics.py (wenet_compute_cer.py port): upper + strip punct."""
    return " ".join(PUNCT.sub(" ", text.upper()).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)

    processor = build_processor()
    model = build_model().to(device)  # FP32
    report = init_report(model)
    print(f"params: {report['params_total'] / 1e6:.1f}M, frozen: {report['frozen'] or 'none'}")

    sample = fetch_smoke_subset(n=8)[args.sample_index]
    ref_text = sample["text"]
    print(f"sample {sample['id']}: {len(sample['audio_array']) / sample['sampling_rate']:.1f}s")
    print(f"ref: {ref_text}")

    example = prepare_example(sample, processor)
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)
    batch = {k: v.to(device) for k, v in collator([example]).items()}
    print(f"input_features {tuple(batch['input_features'].shape)}, labels {tuple(batch['labels'].shape)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    start = time.time()
    final_loss = None
    for step in range(1, args.steps + 1):
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        final_loss = loss.item()
        if step == 1 or step % 20 == 0:
            print(f"step {step:4d}  loss {final_loss:.4f}")
        if final_loss < 0.01:
            print(f"early stop at step {step}, loss {final_loss:.4f}")
            break

    model.eval()
    with torch.no_grad():
        pred_ids = model.generate(batch["input_features"], max_new_tokens=200)
    hyp_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"hyp: {hyp_text}")

    match = rough_normalize(hyp_text) == rough_normalize(ref_text)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"elapsed {time.time() - start:.0f}s, peak VRAM {peak_gb:.1f}G")
    print(f"loss<0.1: {final_loss < 0.1}  decode match: {match}")
    print("SMOKE ROUND 1: " + ("PASS" if final_loss < 0.1 and match else "FAIL"))


if __name__ == "__main__":
    main()
