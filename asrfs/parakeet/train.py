import argparse
from pathlib import Path

import yaml

from asrfs.common.ctc import build_ctc_trainer
from asrfs.parakeet.dataset import build_collator, build_dataset, ctc_greedy_decode
from asrfs.parakeet.model import build_model, build_processor, save_checkpoint


def build_trainer(cfg, model, processor, train_ds, eval_ds, collator, overrides: dict):
    return build_ctc_trainer(
        cfg, model, processor, train_ds, eval_ds, collator, overrides,
        default_run_name="parakeet",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="asrfs/parakeet/config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size

    processor = build_processor(cfg)
    model = build_model(cfg)
    train_ds, eval_ds = build_dataset(cfg, processor, mode="mini100")
    collator = build_collator(cfg, processor, model)
    trainer = build_trainer(cfg, model, processor, train_ds, eval_ds, collator, overrides)
    trainer.train()

    out = Path(f"outputs/{cfg['run_name']}/final")
    save_checkpoint(model, processor, str(out))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
