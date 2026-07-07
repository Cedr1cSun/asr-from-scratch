"""manifest 数据线 contract 测试(spec 2026-07-07-manifest-loader-design):
source 解析 / md5 指纹 / jsonl 流 / prepare-load 分发 / stale / fail-fast。全部离线。"""

import copy
import json

import numpy as np
import pytest
import soundfile as sf

from asrfs.common import full_data

CFG = {
    "model_size": "fake",
    "run_name": "full_data_unit",
    "data": {"n_train": 100, "n_eval": 20, "max_label_len": 5, "max_audio_seconds": 30.0},
    "training": {"learning_rate": 1.0e-4, "max_steps": 300},
}

# Task 1 Step 1 在 master 60a59fd 捕获的 params_hash(CFG) 输出:
# 锁"hf 源 hash 字节级不变"(spec §三,已算 HF 特征不判 stale)。
HF_HASH_ANCHOR = "9dcd76e3d9d3381a9c982ffe7562ad58dab1df846ccc69ec1d64b6c92f1b40f2"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """本套件语义全依赖 env 缺省;挡住外部 shell 泄漏的覆盖变量。"""
    monkeypatch.delenv("ASRFS_DATA_SOURCE", raising=False)
    monkeypatch.delenv("ASRFS_MANIFEST_PATH", raising=False)


def _write_manifest(tmp_path, rows, name="m.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_hf_hash_byte_anchor():
    assert full_data.params_hash(CFG) == HF_HASH_ANCHOR


def test_source_default_and_explicit_hf_equal():
    explicit = copy.deepcopy(CFG)
    explicit["data"]["source"] = "hf"
    assert full_data.params_hash(explicit) == full_data.params_hash(CFG)


def test_manifest_hash_uses_content_not_path(tmp_path):
    rows = [{"path": "/x/a.wav", "target": "OK", "task": "ASR"}]
    p1 = _write_manifest(tmp_path, rows)
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    cfg["data"]["manifest_path"] = str(p1)
    h1 = full_data.params_hash(cfg)
    assert h1 != full_data.params_hash(CFG)  # manifest 线与 hf 线指纹不同

    p2 = tmp_path / "copy.jsonl"
    p2.write_bytes(p1.read_bytes())          # 同内容异路径 → hash 不变
    cfg2 = copy.deepcopy(cfg)
    cfg2["data"]["manifest_path"] = str(p2)
    assert full_data.params_hash(cfg2) == h1

    p1.write_text(p1.read_text().replace("OK", "NO"))  # 内容变 → hash 变
    assert full_data.params_hash(cfg) != h1


def test_manifest_hash_missing_file(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    cfg["data"]["manifest_path"] = str(tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        full_data.params_hash(cfg)


def test_manifest_source_requires_path():
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    with pytest.raises(ValueError, match="manifest_path"):
        full_data.params_hash(cfg)


def test_invalid_source_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "s3"
    with pytest.raises(ValueError, match="source"):
        full_data.params_hash(cfg)


def test_env_overrides(monkeypatch, tmp_path):
    p = _write_manifest(tmp_path, [{"path": "/x/a.wav", "target": "OK"}])
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    cfg["data"]["manifest_path"] = str(p)

    # env source 覆盖 cfg:强制 hf → 回落 hf 锚
    monkeypatch.setenv("ASRFS_DATA_SOURCE", "hf")
    assert full_data.params_hash(cfg) == HF_HASH_ANCHOR
    monkeypatch.delenv("ASRFS_DATA_SOURCE")

    # env path 覆盖 cfg path:指向不同内容 → hash 变
    other = _write_manifest(tmp_path, [{"path": "/y/b.wav", "target": "DIFFERENT"}], name="other.jsonl")
    h_cfg_path = full_data.params_hash(cfg)
    monkeypatch.setenv("ASRFS_MANIFEST_PATH", str(other))
    assert full_data.params_hash(cfg) != h_cfg_path


def _write_wav(tmp_path, name, seconds=1.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / name
    sf.write(str(p), wav, sr)
    return p, len(wav)


def test_stream_manifest_row_contract(tmp_path):
    wav_path, n = _write_wav(tmp_path, "utt-001.wav")
    mp = _write_manifest(tmp_path, [{"path": str(wav_path), "target": "HELLO WORLD", "task": "ASR"}])
    rows = list(full_data._stream_manifest(mp))
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == "utt-001"                       # wav 文件名去扩展名
    assert r["text"] == "HELLO WORLD"                 # target 原样(全大写不动)
    assert r["sampling_rate"] == 16000
    assert r["audio_array"].dtype == np.float32 and len(r["audio_array"]) == n


def test_stream_manifest_subset_head(tmp_path):
    wav_path, _ = _write_wav(tmp_path, "u.wav")
    mp = _write_manifest(tmp_path, [{"path": str(wav_path), "target": f"T{i}"} for i in range(5)])
    out = list(full_data._stream_manifest(mp, subset_head=2))
    assert [r["text"] for r in out] == ["T0", "T1"]


def test_stream_manifest_missing_wav(tmp_path):
    mp = _write_manifest(tmp_path, [{"path": str(tmp_path / "ghost.wav"), "target": "X"}])
    with pytest.raises(FileNotFoundError, match="ghost.wav"):
        list(full_data._stream_manifest(mp))


def test_stream_manifest_bad_rows(tmp_path):
    wav_path, _ = _write_wav(tmp_path, "u.wav")
    good = json.dumps({"path": str(wav_path), "target": "OK"})
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text(good + "\nnot-json\n")
    with pytest.raises(ValueError, match=r":2:"):     # 1-based 行号
        list(full_data._stream_manifest(bad_json))

    missing_field = tmp_path / "missing.jsonl"
    missing_field.write_text(json.dumps({"path": str(wav_path)}) + "\n")  # 缺 target
    with pytest.raises(ValueError, match=r":1:"):
        list(full_data._stream_manifest(missing_field))


class FakeAdapter:
    __name__ = "asrfs.manifesttest"

    @staticmethod
    def build_processor(cfg):
        return "fake-processor"

    @staticmethod
    def make_example(processor, audio, sampling_rate, text):
        assert processor == "fake-processor"
        frames = max(1, len(audio) // 1600)
        feats = np.full((frames, 2), float(len(audio)), dtype=np.float32)
        return {"input_features": feats, "labels": [ord(c) % 32 for c in text.lower()]}


@pytest.fixture()
def manifest_env(monkeypatch, tmp_path):
    """manifest 源 prepare/load 环境:数据目录隔离 + eval 仍走(fake 的)HF 流。"""
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path / "data"))

    def _fake_eval_stream(config, split, subset_head=None):
        sr = 16000
        yield {"id": f"{config}.{split}-eval", "audio_array": np.zeros(sr, dtype=np.float32),
               "sampling_rate": sr, "text": "ok"}

    monkeypatch.setattr(full_data, "_stream_split", _fake_eval_stream)
    return tmp_path


def _manifest_cfg(tmp_path, n_utts=2, speed_perturb=None):
    wavs = [_write_wav(tmp_path, f"utt-{i:03d}.wav")[0] for i in range(n_utts)]
    mp = _write_manifest(tmp_path, [{"path": str(w), "target": "OK", "task": "ASR"} for w in wavs])
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    cfg["data"]["manifest_path"] = str(mp)
    if speed_perturb:
        cfg["data"]["speed_perturb"] = speed_perturb
    return cfg, mp


def test_prepare_manifest_source_roundtrip(manifest_env):
    cfg, _ = _manifest_cfg(manifest_env, n_utts=2)
    manifest = full_data.prepare_full_dataset(cfg, FakeAdapter)
    # manifest 源:train 单 split "train.960" + eval 恒 validation.clean(spec §一)
    assert sorted(manifest["splits"]) == ["train.960", "validation.clean"]
    assert manifest["splits"]["train.960"] == {"rows_before": 2, "rows_after": 2}
    assert manifest["splits"]["validation.clean"] == {"rows_before": 1, "rows_after": 1}

    train, eval_ds = full_data.load_full_dataset(cfg, model_name="manifesttest")
    assert len(train) == 2 and len(eval_ds) == 1
    assert set(train.column_names) >= {"input_features", "labels", "length"}


def test_prepare_manifest_speed_perturb_triples(manifest_env):
    # 变速管线零改动的回归锚:train ×3、eval 不变速(与 test_full_data 的 hf 版同构)
    cfg, _ = _manifest_cfg(manifest_env, n_utts=1, speed_perturb=[0.9, 1.0, 1.1])
    manifest = full_data.prepare_full_dataset(cfg, FakeAdapter)
    assert manifest["splits"]["train.960"]["rows_after"] == 3
    assert manifest["splits"]["validation.clean"]["rows_after"] == 1


def test_load_manifest_source_stale_on_content_change(manifest_env):
    cfg, mp = _manifest_cfg(manifest_env, n_utts=1)
    full_data.prepare_full_dataset(cfg, FakeAdapter)
    mp.write_text(mp.read_text().replace("OK", "NO"))   # 内容变 → md5 变
    with pytest.raises(ValueError, match="stale"):
        full_data.load_full_dataset(cfg, model_name="manifesttest")


def test_prepare_manifest_missing_file_fails_before_work(manifest_env):
    # hash 前置(spec §三):jsonl 缺失须在写任何 split 之前抛,不能白算几小时
    cfg = copy.deepcopy(CFG)
    cfg["data"]["source"] = "manifest"
    cfg["data"]["manifest_path"] = str(manifest_env / "nope.jsonl")
    with pytest.raises(FileNotFoundError):
        full_data.prepare_full_dataset(cfg, FakeAdapter)
    assert not (manifest_env / "data" / "full" / "manifesttest").exists()


def test_hf_source_dispatch_unchanged(manifest_env):
    # source 缺省 → hf:仍产既有三 train split 布局(向后兼容锚)
    manifest = full_data.prepare_full_dataset(CFG, FakeAdapter)
    assert sorted(manifest["splits"]) == sorted(
        ["train.clean.100", "train.clean.360", "train.other.500", "validation.clean"]
    )
