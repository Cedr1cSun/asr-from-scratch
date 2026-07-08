"""离线整 split 评测(Table 3 交付用):
    python eval_full.py <model> <ckpt_dir> <split> [--limit N] [--batch B] [--device D]
split ∈ validation.clean | test.clean | test.other
特征在线抽(adapter make_example),不依赖预计算;打分走 asrfs.common.metrics.wer
(wenet 归一化口径)。参考文本 = HF 原始 transcript。
在仓根目录下跑;集群上需 export HF_ENDPOINT=https://hf-mirror.com。
"""

import argparse
import json
import time
from importlib import import_module

import torch
import yaml

SPLITS = {
    "validation.clean": ("clean", "validation"),
    "test.clean": ("clean", "test"),
    "test.other": ("other", "test"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["whisper", "parakeet", "sensevoice", "x_asr"])
    ap.add_argument("ckpt")
    ap.add_argument("split", choices=sorted(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from asrfs.common.full_data import _stream_split
    from asrfs.common.metrics import wer

    adapter = import_module(f"asrfs.{a.model}")
    cfg = yaml.safe_load(open(f"asrfs/{a.model}/config_full.yaml"))
    try:
        model, processor = adapter.load_checkpoint(cfg, a.ckpt)
    except Exception as e:
        # 中途 checkpoint-N 只有裸模型(save_checkpoint 只在训练末尾写 tokenizer/spm);
        # 回退:结构从 cfg 建 + safetensors 权重,processor 从仓内钉死资产建(与训练同源)。
        print(
            f"load_checkpoint failed ({type(e).__name__}); "
            "fallback: build_model(cfg)+safetensors+build_processor(cfg)",
            flush=True,
        )
        from safetensors.torch import load_file

        model = adapter.build_model(cfg)
        model.load_state_dict(load_file(f"{a.ckpt}/model.safetensors"))
        processor = adapter.build_processor(cfg)
    if a.device:
        dev = a.device
    elif torch.cuda.device_count() > 1:
        dev = "cuda:1"
    elif torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    model.eval().to(dev)
    coll = adapter.build_collator(cfg, processor, model)
    config, split = SPLITS[a.split]

    refs: list[str] = []
    hyps: list[str] = []
    buf_ex: list[dict] = []
    buf_ref: list[str] = []
    t0 = time.time()
    n = 0

    def flush() -> None:
        nonlocal buf_ex, buf_ref
        if not buf_ex:
            return
        batch = coll(buf_ex)
        out = adapter.decode(model, processor, batch)
        hyps.extend(out)
        refs.extend(buf_ref)
        buf_ex, buf_ref = [], []

    for row in _stream_split(config, split, subset_head=a.limit):
        buf_ex.append(
            adapter.make_example(processor, row["audio_array"], row["sampling_rate"], row["text"])
        )
        buf_ref.append(row["text"])
        n += 1
        if len(buf_ex) >= a.batch:
            flush()
            if (n // a.batch) % 20 == 0:
                print(
                    f"[{n}] running_wer={wer(refs, hyps):.4f} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    flush()

    result = {
        "model": a.model,
        "ckpt": a.ckpt,
        "split": a.split,
        "n_utts": len(refs),
        "wer": round(wer(refs, hyps), 4),
        "seconds": round(time.time() - t0, 1),
    }
    print("RESULT " + json.dumps(result))
    for i in range(min(3, len(refs))):
        print(f"REF: {refs[i][:80]}")
        print(f"HYP: {hyps[i][:80]!r}")


if __name__ == "__main__":
    main()
