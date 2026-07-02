import argparse
from pathlib import Path

import yaml

from asrfs.whisper.dataset import build_collator, build_dataset
from asrfs.whisper.model import build_model, build_processor, save_checkpoint
from asrfs.whisper.trainer import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--max-steps", type=int, default=None, help="override config, for dry runs")
    parser.add_argument("--lr", type=float, default=None, help="override config peak learning rate")
    parser.add_argument("--run-name", type=str, default=None, help="override config run_name")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

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
