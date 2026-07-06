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

    rng 参数仅测试注入用;生产路径每个 dataloader worker 进程各持一个
    default_rng(OS 熵种子),互不相关。
    """
    params = SpecAugmentParams(**dict(aug_cfg["spec_augment"]))
    state = {"rng": rng or np.random.default_rng(), "logged": False}

    def transform(batch: dict) -> dict:
        if not state["logged"]:
            logger.warning("SpecAugment active: %s", params)
            state["logged"] = True
        out_feats = []
        for feat, length in zip(batch["input_features"], batch["length"]):
            arr = np.asarray(feat, dtype=np.float16)
            out_feats.append(
                spec_augment_single(arr, params, int(length), state["rng"])
            )
        batch["input_features"] = out_feats
        return batch

    return transform
