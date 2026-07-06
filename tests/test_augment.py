import numpy as np
import pytest

from asrfs.common.augment import (
    SpecAugmentParams,
    build_spec_augment_transform,
    real_frames,
    spec_augment_single,
)

AUG_CFG = {
    "spec_augment": {
        "time_axis": 0,
        "num_feature_masks": 2,
        "features_mask_size": 27,
        "num_frame_masks": 10,
        "frames_mask_size": 100,
        "max_frames_mask_fraction": 0.15,
        "p": 0.9,
    }
}


def _params(**over):
    kw = dict(AUG_CFG["spec_augment"])
    kw.update(over)
    return SpecAugmentParams(**kw)


def test_real_frames_formula_and_cap():
    # 16k * 2s = 32000 样本 → 32000//160 + 1 = 201 帧
    assert real_frames(32000, 3000) == 201
    # 上限钳到特征实际 T
    assert real_frames(32000, 150) == 150


def test_mask_confined_to_real_frames_time_major():
    # (T=500, F=80),真实帧 201,padding 区必须逐位不动
    rng = np.random.default_rng(0)
    feat = np.zeros((500, 80), dtype=np.float16)
    feat[:201] = np.linspace(6.9, 7.1, 201 * 80, dtype=np.float16).reshape(201, 80)
    out = spec_augment_single(feat, _params(p=1.0), length_samples=32000, rng=rng)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out[201:], feat[201:])
    assert not np.array_equal(out[:201], feat[:201])  # p=1 必然动了真实区


def test_mask_confined_freq_major():
    # whisper 轴向 (F=80, T=3000)
    rng = np.random.default_rng(0)
    feat = np.zeros((80, 3000), dtype=np.float16)
    feat[:, :201] = np.linspace(6.9, 7.1, 80 * 201, dtype=np.float16).reshape(80, 201)
    out = spec_augment_single(feat, _params(time_axis=1, p=1.0), length_samples=32000, rng=rng)
    np.testing.assert_array_equal(out[:, 201:], feat[:, 201:])
    assert not np.array_equal(out[:, :201], feat[:, :201])


def test_input_not_mutated():
    rng = np.random.default_rng(0)
    feat = np.full((500, 80), 7.0, dtype=np.float16)
    snapshot = feat.copy()
    spec_augment_single(feat, _params(p=1.0), length_samples=32000, rng=rng)
    np.testing.assert_array_equal(feat, snapshot)


def test_fill_value_is_real_region_mean():
    # 非常值特征:mask 填充值必须等于「真实帧区间」的 mean(常值特征下 mean 填充是
    # no-op,无法区分,故必须用非常值;padding 区全 0 若被抄进 mean 会立刻偏离)
    rng = np.random.default_rng(2)
    feat2 = np.zeros((500, 80), dtype=np.float16)
    feat2[:201] = np.linspace(0, 10, 201 * 80, dtype=np.float16).reshape(201, 80)
    expected_mean = np.float16(feat2[:201].astype(np.float32).mean())
    out2 = spec_augment_single(feat2, _params(p=1.0), length_samples=32000, rng=rng)
    np.testing.assert_array_equal(out2[201:], feat2[201:])  # padding 区不动
    diff_mask = out2[:201] != feat2[:201]
    assert diff_mask.any()
    filled = out2[:201][diff_mask]
    np.testing.assert_array_equal(filled, np.full_like(filled, expected_mean))


def test_time_mask_budget_respected():
    # max_frames_mask_fraction=0.15:time-mask 整行填充帧数 ≤ ceil(0.15*201)=31
    # (实现推导:num=min(10,ceil(30.15/100))=1,max_w=30,integers 高端开 → 实际 ≤29)
    rng = np.random.default_rng(3)
    feat = np.linspace(0, 10, 500 * 80, dtype=np.float16).reshape(500, 80)
    p = _params(p=1.0, num_feature_masks=0)  # 只留 time mask,便于数行
    out = spec_augment_single(feat, p, length_samples=32000, rng=rng)
    masked_rows = int((out[:201] != feat[:201]).all(axis=1).sum())
    assert masked_rows <= int(np.ceil(0.15 * 201))


def test_freq_mask_count_and_width():
    # 只留 freq mask:任意行被改动的列构成 ≤2 个连续区段,每段宽 ≤26(integers(0,27) 高端开)
    rng = np.random.default_rng(6)
    feat = np.linspace(0, 10, 500 * 80, dtype=np.float16).reshape(500, 80)
    p = _params(p=1.0, num_frame_masks=0)
    out = spec_augment_single(feat, p, length_samples=32000, rng=rng)
    changed_cols = np.where((out[:201] != feat[:201]).any(axis=0))[0]
    assert changed_cols.size > 0
    # 数连续区段
    segments = int((np.diff(changed_cols) > 1).sum()) + 1
    assert segments <= 2
    widths = np.split(changed_cols, np.where(np.diff(changed_cols) > 1)[0] + 1)
    assert all(len(w) <= 26 for w in widths)


def test_p_zero_is_identity():
    rng = np.random.default_rng(0)
    feat = np.linspace(0, 10, 500 * 80, dtype=np.float16).reshape(500, 80)
    out = spec_augment_single(feat, _params(p=0.0), length_samples=32000, rng=rng)
    np.testing.assert_array_equal(out, feat)


def test_transform_batch_and_log(caplog):
    tf = build_spec_augment_transform(
        {"spec_augment": {**AUG_CFG["spec_augment"], "p": 1.0}},
        rng=np.random.default_rng(0),
    )
    batch = {
        "input_features": [np.full((500, 80), 7.0, dtype=np.float16).tolist()] * 2,
        "length": [32000, 16000],
        "labels": [[1, 2], [3]],
        "id": ["a", "b"],
        "text": ["x", "y"],
    }
    with caplog.at_level("WARNING", logger="asrfs.common.augment"):
        out = tf(batch)
    assert any("SpecAugment active" in r.message for r in caplog.records)
    assert len(out["input_features"]) == 2
    assert np.asarray(out["input_features"][0]).dtype == np.float16
    # 非特征列原样带回
    assert out["labels"] == [[1, 2], [3]] and out["id"] == ["a", "b"]
