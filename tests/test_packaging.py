import importlib

MODULES = [
    "asrfs",
    "asrfs.common.data",
    "asrfs.common.metrics",
    "asrfs.whisper.model",
    "asrfs.whisper.dataset",
    "asrfs.whisper.train",
    "asrfs.whisper.smoke",
    "asrfs.whisper.batch_probe",
    "asrfs.whisper.reload_check",
    "asrfs.parakeet.model",
    "asrfs.parakeet.dataset",
    "asrfs.parakeet.train",
    "asrfs.parakeet.smoke",
    "asrfs.parakeet.reload_check",
]

def test_asrfs_modules_importable():
    for name in MODULES:
        importlib.import_module(name)

def test_data_dir_anchors_repo_root():
    from asrfs.common import data

    assert (data.PROJECT_ROOT / "requirements.txt").is_file()
    assert data.DATA_DIR == data.PROJECT_ROOT / "data"
