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
