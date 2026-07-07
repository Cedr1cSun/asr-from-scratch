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
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
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
# manifest 源 train 整体一个 split(jsonl 本就是拼好的 960h,不按 100/360/500 拆,
# 避免解析路径子串的脆弱逻辑);eval 恒走 HF validation.clean,不随 source 分叉。
MANIFEST_TRAIN_SPLIT_NAME = "train.960"

FEATURE_DTYPE = "float16"
DEFAULT_MAX_AUDIO_SECONDS = 30.0
_NON_FEATURE_CFG_KEYS = ("data", "training", "smoke", "run_name", "augment")
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


DEFAULT_SPEED_PERTURB = (1.0,)


def _normalized_speed_perturb(data_cfg: dict) -> list[float]:
    """hash 语义(spec §2):缺键 ≡ 显式 [1.0]。"""
    raw = data_cfg.get("speed_perturb")
    if raw is None:
        raw = list(DEFAULT_SPEED_PERTURB)
    return [float(x) for x in raw]


def _tokenizer_fingerprint(model_name: str, cfg: dict) -> str | None:
    """适配包声明的 tokenizer/FE 字节身份(round-2 审计 F1/F2)。

    labels 与特征由 tokenizer/feature extractor 决定,但它们活在代码里而非 cfg,
    单靠 cfg 子集哈希抓不到"重训 BPE / Hub 更新 tokenizer"这类变更 → 旧缓存会被
    静默复用。各 asrfs.<model>.model 暴露 tokenizer_fingerprint(cfg) -> str
    (x_asr:spm 文件 sha256;其余:source@pinned-revision)。未注册模型(单测
    FakeAdapter 等)返回 None——键仍写入 hash,语义稳定。惰性 import,无环。
    """
    try:
        mod = importlib.import_module(f"asrfs.{model_name}.model")
    except ModuleNotFoundError:
        return None
    fp = getattr(mod, "tokenizer_fingerprint", None)
    return fp(cfg) if callable(fp) else None


def _resolve_source(data_cfg: dict) -> str:
    """数据源开关(spec 2026-07-07 §二):env ASRFS_DATA_SOURCE > cfg data.source > 缺省 hf。
    缺省即 hf 保证现有 config/测试零改动;本地 e2e 对 config_full 用 env 强制 hf。"""
    src = os.environ.get("ASRFS_DATA_SOURCE") or data_cfg.get("source") or "hf"
    if src not in ("hf", "manifest"):
        raise ValueError(f"data.source must be 'hf' or 'manifest', got {src!r}")
    return src


def _resolve_manifest_path(data_cfg: dict) -> Path:
    """manifest 路径:env ASRFS_MANIFEST_PATH > cfg data.manifest_path;缺失即抛,不拖到读数据。"""
    raw = os.environ.get("ASRFS_MANIFEST_PATH") or data_cfg.get("manifest_path")
    if not raw:
        raise ValueError(
            "data.source=manifest requires data.manifest_path in cfg or ASRFS_MANIFEST_PATH env"
        )
    return Path(raw)


def _manifest_md5(path: Path) -> str:
    """jsonl 整文件流式 md5(104 MB 秒级)。内容即数据集身份:改内容才重算特征,挪路径不重算。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_hash(cfg: dict, tokenizer_fingerprint: str | None = None) -> str:
    """sha256(排序后的特征相关 cfg 子集):model 段 + 过滤参数 + 数据 revision + dtype
    + tokenizer 指纹。

    training/smoke/run_name 与 data.n_train/n_eval 不影响预计算特征,排除在外,
    避免调训练超参就把 60 GB 特征判失效;model.gradient_checkpointing/
    generation_max_length/apply_spec_augment 同理(见 _NON_FEATURE_MODEL_KEYS)。
    augment 段是训练期特征增广(datasets set_transform 挂接,惰性、非预计算),不
    改变磁盘上的特征字节,同样排除;data.speed_perturb 是预计算期变速增广(train
    split 三态 0.9/1.0/1.1,见 prepare_full_dataset/_perturb_speed),改变特征字节,
    故纳入 hash(缺键按显式 [1.0] 归一化)。tokenizer_fingerprint 见
    _tokenizer_fingerprint——prepare/load 双侧都传,保证"换 tokenizer ⇒ 判 stale"。
    data.source=manifest 时数据集身份键换为 manifest_md5(jsonl 整文件 md5),见 _resolve_source/_manifest_md5。
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
        "speed_perturb": _normalized_speed_perturb(data_cfg),
        "feature_dtype": FEATURE_DTYPE,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
    # 数据集身份按 source 分叉(spec 2026-07-07 §三):hf 沿用原键原值(json.dumps
    # sort_keys 下与改动前字节级一致,已算 HF 特征不判 stale);manifest 用 jsonl 内容
    # md5。键名不同,两线指纹不可能相撞;manifest 文件缺失在此处即抛 FileNotFoundError。
    if _resolve_source(data_cfg) == "manifest":
        subset["manifest_md5"] = _manifest_md5(_resolve_manifest_path(data_cfg))
    else:
        subset["librispeech_revision"] = LIBRISPEECH_REVISION
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


def _stream_manifest(manifest_path, subset_head: int | None = None):
    """公司 jsonl 清单流(spec 2026-07-07 §一):逐行 {path, target} → soundfile 读集群
    wav,产出与 _stream_split 同构的行。全部 fail fast(spec §四):静默跳行会让实际
    数据与 manifest_md5 指纹脱钩,公司池缺文件是异常必须暴露。测试直接喂 tmp jsonl。"""
    manifest_path = Path(manifest_path)
    yielded = 0
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if subset_head is not None and yielded >= subset_head:
                return
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{manifest_path}:{lineno}: invalid JSON: {e}") from e
            if "path" not in row or "target" not in row:
                raise ValueError(f"{manifest_path}:{lineno}: row missing 'path' or 'target'")
            wav_path = row["path"]
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(f"{manifest_path}:{lineno}: wav not found: {wav_path}")
            array, sampling_rate = sf.read(wav_path, dtype="float32")
            yielded += 1
            yield {
                "id": Path(wav_path).stem,
                "audio_array": np.asarray(array, dtype=np.float32),
                "sampling_rate": int(sampling_rate),
                "text": row["target"],
            }


def _split_plan(cfg: dict, subset_head: int | None):
    """产 (split_name, factors, raw_rows_factory) 序列;source 分叉集中在这一处。"""
    data_cfg = cfg.get("data") or {}
    speed_factors = _normalized_speed_perturb(data_cfg)
    plan = []
    if _resolve_source(data_cfg) == "manifest":
        mpath = _resolve_manifest_path(data_cfg)
        plan.append(
            (MANIFEST_TRAIN_SPLIT_NAME, speed_factors, lambda: _stream_manifest(mpath, subset_head))
        )
    else:
        for config, split, split_name in TRAIN_SPLITS:
            plan.append(
                (split_name, speed_factors,
                 lambda config=config, split=split: _stream_split(config, split, subset_head))
            )
    config, split, split_name = EVAL_SPLIT
    plan.append((split_name, [1.0], lambda: _stream_split(config, split, subset_head)))
    return plan


def _perturb_speed(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """torchaudio sox-speed 语义(变速变调);factor=1.0 恒等返回原 array。
    0.9 → 变慢变长,1.1 → 变快变短(icefall/lhotse 三态增广)。"""
    if factor == 1.0:
        return audio
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    out, _ = AF.speed(wav, orig_freq=sr, factor=factor)
    return out.numpy()


def _prepared_rows(raw_rows, adapter, processor, cfg, counters, speed_factors):
    data_cfg = cfg.get("data") or {}
    max_audio_s = float(data_cfg.get("max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS))
    max_label_len = data_cfg.get("max_label_len")
    for row in raw_rows:
        counters["rows_before"] += 1
        for factor in speed_factors:
            audio = _perturb_speed(row["audio_array"], row["sampling_rate"], factor)
            if len(audio) / row["sampling_rate"] > max_audio_s:  # 变速后时长过滤
                continue
            example = adapter.make_example(processor, audio, row["sampling_rate"], row["text"])
            if max_label_len is not None and len(example["labels"]) > int(max_label_len):
                continue
            example["input_features"] = np.asarray(example["input_features"], dtype=np.float16)
            example["length"] = len(audio)  # 变速后采样点数
            counters["rows_after"] += 1
            yield example


def prepare_full_dataset(cfg: dict, processor_adapter, subset_head: int | None = None) -> dict:
    adapter = processor_adapter
    processor = adapter.build_processor(cfg)
    model_name = model_name_of(adapter)
    tokenizer_fp = _tokenizer_fingerprint(model_name, cfg)
    # hash 前置(spec §三):manifest 漏配路径/文件缺失在任何重活前即刻抛;md5 只读一次,
    # manifest.json 记录的是本次预计算实际消费的清单内容快照。
    cfg_hash = params_hash(cfg, tokenizer_fingerprint=tokenizer_fp)
    out_root = full_dir(model_name)
    out_root.mkdir(parents=True, exist_ok=True)

    # 不变量:manifest.json 存在 ⇒ 磁盘上所有 split 均已重写完成且与其 hash 一致。
    # 先删旧 manifest 再跑 split,这样中途崩溃的重跑不会留下一份指向新旧混杂特征的旧
    # manifest;manifest 仍然只在全部 split 成功后于末尾写入(见下方),中断的重跑会
    # 在 load_full_dataset 里经既有 FileNotFoundError 兜底报错。
    (out_root / "manifest.json").unlink(missing_ok=True)

    manifest_splits = {}
    for split_name, factors, raw_factory in _split_plan(cfg, subset_head):
        counters = {"rows_before": 0, "rows_after": 0}
        split_dir = out_root / split_name
        work_dir = out_root / f".work_{split_name}"
        if work_dir.exists():
            shutil.rmtree(work_dir)

        def gen(factors=factors, raw_factory=raw_factory):  # 默认参数绑定,避免闭包 late-binding
            yield from _prepared_rows(raw_factory(), adapter, processor, cfg, counters, factors)

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
        "params_hash": cfg_hash,
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
    expected = params_hash(cfg, tokenizer_fingerprint=_tokenizer_fingerprint(model_name, cfg))
    if manifest["params_hash"] != expected:
        raise ValueError(
            f"stale full-mode features for {model_name}: manifest params_hash "
            f"{manifest['params_hash'][:12]}.. != cfg {expected[:12]}..; "
            f"re-run prepare_full_dataset"
        )
    if manifest.get("subset_head") is not None and os.environ.get("ASRFS_ALLOW_SUBSET") != "1":
        raise RuntimeError(
            f"{model_name}: full features were precomputed with subset_head="
            f"{manifest['subset_head']} (partial data), not the full 960h set. "
            f"Training on these would silently produce a wrong baseline. "
            f"Re-run prepare_full_dataset without --subset-head, or set "
            f"ASRFS_ALLOW_SUBSET=1 to intentionally train on the subset (e2e only)."
        )
    train_names = (
        (MANIFEST_TRAIN_SPLIT_NAME,)
        if _resolve_source(cfg.get("data") or {}) == "manifest"
        else TRAIN_SPLIT_NAMES
    )
    train = concatenate_datasets([load_from_disk(str(root / name)) for name in train_names])
    eval_ds = load_from_disk(str(root / EVAL_SPLIT_NAME))
    aug_cfg = cfg.get("augment")
    if aug_cfg:
        from asrfs.common.augment import build_spec_augment_transform

        # 只挂 train;eval(validation.clean)保持干净(spec §1)
        train.set_transform(build_spec_augment_transform(aug_cfg))
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
    # main() writes every split + manifest.json synchronously and returns only on
    # success; the process is functionally complete here. Python's own interpreter
    # finalization then aborts with "Fatal Python error: PyGILState_Release: thread
    # state ... must be current when releasing" (SIGABRT / exit 134). It is a
    # teardown artifact of a background decode thread (HF datasets streaming +
    # soundfile/torchcodec) not joined before finalize — reproduces even with
    # speed_perturb absent and torch/torchaudio never imported, so it pre-exists
    # this branch and is environment-level (torch 2.12 / datasets 5.0). It corrupts
    # nothing: the crash is strictly after main() returns, manifest is already on
    # disk. But a cluster job runner reads the non-zero exit as a precompute
    # failure, so flush stdout/stderr and hard-exit 0 to report the real outcome
    # and skip the crashing finalization. A genuine failure inside main() raises
    # before this line, keeping a real non-zero exit.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
