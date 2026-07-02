from pathlib import Path

from datasets import Dataset

from asrfs.common import data

def test_env_var_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path / "custom"))
    assert data.data_dir() == tmp_path / "custom"

def test_default_is_repo_root_data(monkeypatch):
    monkeypatch.delenv("ASRFS_DATA_DIR", raising=False)
    assert data.data_dir() == data.DATA_DIR
    assert data.data_dir() == Path(data.__file__).resolve().parent.parent.parent / "data"

def test_fetch_smoke_subset_resolves_cache_under_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path))
    fake = Dataset.from_list(
        [{"id": "fake-0", "audio_array": [0.0, 0.1], "sampling_rate": 16000, "text": "HI"}]
    )
    fake.save_to_disk(str(tmp_path / "smoke_train.clean.100_head1"))
    out = data.fetch_smoke_subset(n=1)
    assert out[0]["id"] == "fake-0"
    assert out[0]["text"] == "HI"
