import argparse
import time

import torch

from asrfs.common.data import fetch_smoke_subset
from asrfs.common.metrics import normalize_tokens
from asrfs.parakeet.dataset import ParakeetCollator, ctc_greedy_decode, prepare_example
from asrfs.parakeet.model import build_feature_extractor, build_model, build_tokenizer, init_report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)

    fe, tok = build_feature_extractor(), build_tokenizer()
    model = build_model({}).to(device)
    report = init_report(model)
    print(f"params: {report['params_total'] / 1e6:.1f}M, frozen: {report['frozen'] or 'none'}")

    sample = fetch_smoke_subset(n=8)[args.sample_index]
    print(f"ref: {sample['text']}")
    example = prepare_example(sample, fe, tok)
    blank = tok.vocab_size
    batch = {k: v.to(device) for k, v in ParakeetCollator(fe, pad_label_id=blank)([example]).items()}
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
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {final_loss:.4f}")
        if final_loss < 0.01:
            print(f"early stop at step {step}, loss {final_loss:.4f}")
            break

    model.eval()
    with torch.no_grad():
        logits = model(
            input_features=batch["input_features"], attention_mask=batch["attention_mask"]
        ).logits
    hyp = ctc_greedy_decode(logits.argmax(-1), tok, blank_id=blank)[0]
    print(f"hyp: {hyp}")
    match = normalize_tokens(hyp) == normalize_tokens(sample["text"])
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"elapsed {time.time() - start:.0f}s, peak VRAM {peak_gb:.1f}G")
    print(f"loss<0.1: {final_loss < 0.1}  decode match: {match}")
    print("SMOKE ROUND 1: " + ("PASS" if final_loss < 0.1 and match else "FAIL"))

if __name__ == "__main__":
    main()
