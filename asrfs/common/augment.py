"""SpecAugment(lhotse 语义移植)— full 模式训练期特征增广(spec 2026-07-06 §1)。

只被 load_full_dataset 的 train split 经 set_transform 挂接;冒烟/标定路径不
import 本模块。与 lhotse.dataset.signal_transforms.SpecAugment 的对齐与差异:
- 对齐:参数默认值;p 按 utterance 门控(random > p 整条跳过);time mask 数与
  宽度的收敛公式 num = min(num_frame_masks, ceil(max_tot/frames_mask_size)),
  max_w = min(frames_mask_size, max_tot // num);mean 填充,freq/time 共用。
- 差异:time warp 不做(spec 记为与 icefall 的已知差异);lhotse 面对未 pad 的
  per-cut 特征,本移植面对可能预 pad 的特征(whisper 恒 (80,3000)),故 mask
  与 mean 都限定在真实帧区间 [0, real_frames)。
"""

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# 四模型特征提取器一致:16 kHz、10 ms hop(spec §1);center=True 下 T = length//160 + 1
HOP = 160


@dataclass(frozen=True)
class SpecAugmentParams:
    time_axis: int
    num_feature_masks: int = 2
    features_mask_size: int = 27
    num_frame_masks: int = 10
    frames_mask_size: int = 100
    max_frames_mask_fraction: float = 0.15
    p: float = 0.9


def real_frames(length_samples: int, total_frames: int) -> int:
    return min(total_frames, length_samples // HOP + 1)


def spec_augment_single(
    feat: np.ndarray,
    params: SpecAugmentParams,
    length_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """单条特征增广;返回新数组,不改入参。轴向由 params.time_axis 指定。"""
    if rng.random() > params.p:
        return feat.copy()
    out = feat.copy()
    tf = out if params.time_axis == 0 else out.T  # (T, F) 视图,写视图即写 out
    t_real = real_frames(length_samples, tf.shape[0])
    if t_real <= 0:
        return out
    region = tf[:t_real]
    mean = np.float16(region.astype(np.float32).mean())

    n_feat = tf.shape[1]
    for _ in range(params.num_feature_masks):
        w = int(rng.integers(0, params.features_mask_size))
        if w == 0 or w >= n_feat:
            continue
        start = int(rng.integers(0, n_feat - w + 1))
        region[:, start : start + w] = mean

    max_tot = params.max_frames_mask_fraction * t_real
    num_masks = min(
        params.num_frame_masks, int(np.ceil(max_tot / params.frames_mask_size))
    )
    if num_masks > 0:
        max_w = int(min(params.frames_mask_size, max_tot // num_masks))
        for _ in range(num_masks):
            w = int(rng.integers(0, max_w)) if max_w > 0 else 0
            if w == 0 or w >= t_real:
                continue
            start = int(rng.integers(0, t_real - w + 1))
            region[start : start + w, :] = mean
    return out


def build_spec_augment_transform(aug_cfg: dict, rng: np.random.Generator | None = None):
    """给 datasets.set_transform 用的 batch transform 工厂(spec §1)。

    rng 按【进程】惰性初始化:不在主进程建 Generator(那样 fork 出的 dataloader
    worker 会逐位复制同一状态,导致各 worker mask 流相同且逐 epoch 重放)。每个
    进程首次调用 transform 时各自取种 —— 注入的 rng 只在建工厂的那个进程沿用
    (保测试确定性),fork 出的 worker 一律用 OS 熵新种,互相独立。

    注入 rng 的 worker 独立性仅在"建工厂的进程先调用过一次 transform 再 fork"时
    成立;若注入了 rng 却在任何调用前就 fork,各子进程首次调用会各自命中"首次"
    分支并复用同一注入 Generator → 流相同。生产路径从不注入 rng(唯一调用点
    full_data.load_full_dataset 不传 rng),该边界不可达;注入仅用于单测。
    """
    params = SpecAugmentParams(**dict(aug_cfg["spec_augment"]))
    state = {"pid": None, "rng": None, "injected": rng, "logged_pid": None}

    def transform(batch: dict) -> dict:
        pid = os.getpid()
        if state["pid"] != pid:
            # 当前进程首次进来:主进程(建工厂的那个)且给了注入 rng → 用注入的;
            # 否则(fork worker,或没注入)→ OS 熵新种,各进程独立。
            use_injected = state["injected"] is not None and state["pid"] is None
            state["rng"] = state["injected"] if use_injected else np.random.default_rng()
            state["pid"] = pid
        if state["logged_pid"] != pid:
            logger.warning("SpecAugment active (pid=%s): %s", pid, params)
            state["logged_pid"] = pid
        out_feats = []
        for feat, length in zip(batch["input_features"], batch["length"]):
            arr = np.asarray(feat, dtype=np.float16)
            out_feats.append(
                spec_augment_single(arr, params, int(length), state["rng"])
            )
        batch["input_features"] = out_feats
        return batch

    return transform
