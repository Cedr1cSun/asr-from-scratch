import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds real models / trainers; deselect with -m 'not slow'"
    )


@pytest.fixture(autouse=True)
def _clean_manifest_env(monkeypatch):
    """final-review F7:manifest-loader 分支后,test_full_data.py 里约 15 处 params_hash
    调用(config 都不带 manifest_path)也经 _resolve_source,一个 shell 里泄漏的
    ASRFS_DATA_SOURCE=manifest/无效值会让它们全部炸 ValueError。仅
    test_manifest_loader.py 有同名 file-local 夹具,对 test_full_data.py/
    test_config_full.py 不生效;挪到这里做成 repo 级 autouse,一次性挡住所有套件。
    test_manifest_loader.py 里的 file-local _clean_env 保留(职责重叠但无害,两者都
    只是 delenv,顺序无关)。"""
    monkeypatch.delenv("ASRFS_DATA_SOURCE", raising=False)
    monkeypatch.delenv("ASRFS_MANIFEST_PATH", raising=False)
