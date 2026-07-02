"""LibriSpeech 960h full-mode 数据管线(设计 spec §3.3)。

prepare_full_dataset 是显式人工/agent 触发的离线预计算步骤(不挂 harness 管线
stage):流式读 train.clean.100 + train.clean.360 + train.other.500(eval 用
validation.clean),逐条调用适配包的 make_example 预计算特征(float16),按模型
分目录 save_to_disk,并写 manifest.json 供 harness training_script stage 校验。

真正的 960h 全量预计算与训练在集群侧执行(每模型特征 ~55-60 GB + HF 下载缓存
~60 GB,单遍数小时到天级);本机(2080 Ti)只用 --subset-head N 做小子集验证。
"""

import argparse
import hashlib
import importlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from asrfs.common.data import LIBRISPEECH_REPO, LIBRISPEECH_REVISION, data_dir

TRAIN_SPLITS = (
    ("clean", "train.100", "train.clean.100"),
    ("clean", "train.360", "train.clean.360"),
    ("other", "train.500", "train.other.500"),
)
EVAL_SPLIT = ("clean", "validation", "validation.clean")
ALL_SPLITS = TRAIN_SPLITS + (EVAL_SPLIT,)
TRAIN_SPLIT_NAMES = tuple(s[2] for s in TRAIN_SPLITS)
EVAL_SPLIT_NAME = EVAL_SPLIT[2]

FEATURE_DTYPE = "float16"
DEFAULT_MAX_AUDIO_SECONDS = 30.0
_NON_FEATURE_CFG_KEYS = ("data", "training", "smoke", "run_name")
# model: 段里这几个 key 是训练/解码期旋钮(见 asrfs/whisper/config.yaml),不改变任何一个
# 预计算出的特征字节;混进 hash 会导致集群上翻一次 gradient_checkpointing 就得重算 60 GB 特征。
_NON_FEATURE_MODEL_KEYS = {"gradient_checkpointing", "generation_max_length", "apply_spec_augment"}


def model_name_of(adapter) -> str:
    # vars()[...] (not plain attribute access): a plain module's __name__ is a
    # normal __dict__ entry, but a class's __name__ goes through a data
    # descriptor on its metaclass that ignores a class-body override (as used
    # by tests/test_full_data.py's FakeAdapter stub) — vars() reads the real
    # entry for both.
    return vars(adapter)["__name__"].rsplit(".", 1)[-1]


def full_dir(model_name: str) -> Path:
    return data_dir() / "full" / model_name


def params_hash(cfg: dict) -> str:
    """sha256(排序后的特征相关 cfg 子集):model 段 + 过滤参数 + 数据 revision + dtype。

    training/smoke/run_name 与 data.n_train/n_eval 不影响预计算特征,排除在外,
    避免调训练超参就把 60 GB 特征判失效;model.gradient_checkpointing/
    generation_max_length/apply_spec_augment 同理(见 _NON_FEATURE_MODEL_KEYS)。
    """
    data_cfg = cfg.get("data") or {}
    model_section = {k: v for k, v in cfg.items() if k not in _NON_FEATURE_CFG_KEYS}
    if isinstance(model_section.get("model"), dict):
        model_section["model"] = {
            k: v for k, v in model_section["model"].items() if k not in _NON_FEATURE_MODEL_KEYS
        }
    subset = {
        "model": model_section,
        "max_label_len": data_cfg.get("max_label_len"),
        "max_audio_seconds": float(data_cfg.get("max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)),
        "librispeech_revision": LIBRISPEECH_REVISION,
        "feature_dtype": FEATURE_DTYPE,
    }
    payload = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stream_split(config: str, split: str, subset_head: int | None = None):
    """流式产出 {id, audio_array, sampling_rate, text};测试 monkeypatch 此函数断网。"""
    stream = load_dataset(
        LIBRISPEECH_REPO,
        config,
        split=split,
        streaming=True,
        revision=LIBRISPEECH_REVISION,
    )
    stream = stream.cast_column("audio", Audio(decode=False))  # 绕开 torchcodec,soundfile 直解
    if subset_head is not None:
        stream = stream.take(subset_head)
    for sample in stream:
        array, sampling_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        yield {
            "id": sample["id"],
            "audio_array": np.asarray(array, dtype=np.float32),
            "sampling_rate": int(sampling_rate),
            "text": sample["text"],
        }


def _prepared_rows(raw_rows, adapter, processor, cfg, counters):
    data_cfg = cfg.get("data") or {}
    max_audio_s = float(data_cfg.get("max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS))
    max_label_len = data_cfg.get("max_label_len")
    for row in raw_rows:
        counters["rows_before"] += 1
        if len(row["audio_array"]) / row["sampling_rate"] > max_audio_s:
            continue
        example = adapter.make_example(
            processor, row["audio_array"], row["sampling_rate"], row["text"]
        )
        if max_label_len is not None and len(example["labels"]) > int(max_label_len):
            continue
        example["input_features"] = np.asarray(example["input_features"], dtype=np.float16)
        example["length"] = len(row["audio_array"])  # 供全量 Trainer group_by_length 分桶
        counters["rows_after"] += 1
        yield example


def prepare_full_dataset(cfg: dict, processor_adapter, subset_head: int | None = None) -> dict:
    adapter = processor_adapter
    processor = adapter.build_processor(cfg)
    out_root = full_dir(model_name_of(adapter))
    out_root.mkdir(parents=True, exist_ok=True)

    # 不变量:manifest.json 存在 ⇒ 磁盘上所有 split 均已重写完成且与其 hash 一致。
    # 先删旧 manifest 再跑 split,这样中途崩溃的重跑不会留下一份指向新旧混杂特征的旧
    # manifest;manifest 仍然只在全部 split 成功后于末尾写入(见下方),中断的重跑会
    # 在 load_full_dataset 里经既有 FileNotFoundError 兜底报错。
    (out_root / "manifest.json").unlink(missing_ok=True)

    manifest_splits = {}
    for config, split, split_name in ALL_SPLITS:
        counters = {"rows_before": 0, "rows_after": 0}
        split_dir = out_root / split_name
        work_dir = out_root / f".work_{split_name}"
        if work_dir.exists():
            shutil.rmtree(work_dir)

        def gen():
            yield from _prepared_rows(
                _stream_split(config, split, subset_head), adapter, processor, cfg, counters
            )

        # from_generator 增量写 arrow,内存不随 split 体量增长(960h 单 split 数十 GB)
        ds = Dataset.from_generator(gen, cache_dir=str(work_dir))
        if split_dir.exists():
            shutil.rmtree(split_dir)
        ds.save_to_disk(str(split_dir))
        del ds
        shutil.rmtree(work_dir)  # 一次性 arrow 工作区:重跑不吃陈旧 cache,行计数恒真实
        manifest_splits[split_name] = counters

    manifest = {
        "splits": manifest_splits,
        "feature_dir": str(out_root),
        "dtype": FEATURE_DTYPE,
        "params_hash": params_hash(cfg),
        "subset_head": subset_head,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_full_dataset(cfg: dict, model_name: str | None = None) -> tuple:
    if model_name is None:
        raise ValueError(
            "load_full_dataset needs model_name (e.g. 'whisper'); "
            "the calling adapter package passes its own name"
        )
    root = full_dir(model_name)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} missing; run: python -m asrfs.common.full_data "
            f"--config <config.yaml> --adapter asrfs.{model_name}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected = params_hash(cfg)
    if manifest["params_hash"] != expected:
        raise ValueError(
            f"stale full-mode features for {model_name}: manifest params_hash "
            f"{manifest['params_hash'][:12]}.. != cfg {expected[:12]}..; "
            f"re-run prepare_full_dataset"
        )
    train = concatenate_datasets(
        [load_from_disk(str(root / name)) for name in TRAIN_SPLIT_NAMES]
    )
    eval_ds = load_from_disk(str(root / EVAL_SPLIT_NAME))
    return train, eval_ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute LibriSpeech full-mode features per adapter package (spec §3.3)"
    )
    parser.add_argument("--config", type=str, required=True, help="adapter config.yaml path")
    parser.add_argument("--adapter", type=str, required=True, help="adapter package, e.g. asrfs.whisper")
    parser.add_argument(
        "--subset-head",
        type=int,
        default=None,
        help="stream only the first N rows per split (local 2080 Ti verification)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    adapter = importlib.import_module(args.adapter)
    manifest = prepare_full_dataset(cfg, adapter, subset_head=args.subset_head)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
