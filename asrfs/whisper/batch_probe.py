import argparse
import time

import torch

from asrfs.common.data import fetch_smoke_subset
from asrfs.whisper.dataset import WhisperCollator, prepare_example
from asrfs.whisper.model import build_model, build_processor

def probe(size: str, grad_checkpoint: bool) -> None:
    device = torch.device("cuda")
    processor = build_processor(size)
    model = build_model(size).to(device)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    example = prepare_example(fetch_smoke_subset(n=8)[0], processor)
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)
    model.train()

    for bs in [1, 2, 4, 8, 16, 32]:
        batch = {k: v.to(device) for k, v in collator([example] * bs).items()}
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                start = time.time()
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.time() - start
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"bs={bs:3d}  step {elapsed:5.2f}s  peak {peak:5.1f}G  ({bs / elapsed:.1f} utt/s)")
        except torch.OutOfMemoryError:
            print(f"bs={bs:3d}  OOM")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="medium", choices=["small", "medium"])
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()
    probe(args.size, args.grad_checkpoint)
