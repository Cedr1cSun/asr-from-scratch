"""full_data 单元测试:manifest 写入 + 过滤,全部走 fake 内存行,零网络。

真实 960h 预计算在集群侧跑(磁盘/时长预算见 asrfs/common/full_data.py 模块
docstring);本套件只锁转换、过滤、manifest、加载语义。
"""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from datasets import load_from_disk

from asrfs.common import full_data

CFG = {
    "model_size": "fake",
    "run_name": "full_data_unit",
    "data": {"n_train": 100, "n_eval": 20, "max_label_len": 5, "max_audio_seconds": 30.0},
    "training": {"learning_rate": 1.0e-4, "max_steps": 300},
}

SPLIT_NAMES = ["train.clean.100", "train.clean.360", "train.other.500", "validation.clean"]


class FakeAdapter:
    __name__ = "asrfs.faketest"

    @staticmethod
    def build_processor(cfg):
        return "fake-processor"

    @staticmethod
    def make_example(processor, audio, sampling_rate, text):
        assert processor == "fake-processor"
        frames = max(1, len(audio) // 1600)
        feats = np.full((frames, 2), float(len(audio)), dtype=np.float32)
        return {"input_features": feats, "labels": [ord(c) % 32 for c in text.lower()]}


def _fake_rows(tag):
    sr = 16000
    return [
        {"id": f"{tag}-keep", "audio_array": np.zeros(sr, dtype=np.float32),
         "sampling_rate": sr, "text": "ok"},
        {"id": f"{tag}-audio-too-long", "audio_array": np.zeros(31 * sr, dtype=np.float32),
         "sampling_rate": sr, "text": "ok"},
        {"id": f"{tag}-label-too-long", "audio_array": np.zeros(sr, dtype=np.float32),
         "sampling_rate": sr, "text": "xxxxxxxxx"},
    ]


def _fake_stream(config, split, subset_head=None):
    rows = _fake_rows(f"{config}.{split}")
    if subset_head is not None:
        rows = rows[:subset_head]
    yield from rows


@pytest.fixture()
def data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ASRFS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(full_data, "_stream_split", _fake_stream)
    return tmp_path


def test_prepare_writes_manifest_and_filters(data_root):
    manifest = full_data.prepare_full_dataset(CFG, FakeAdapter)

    assert sorted(manifest["splits"]) == sorted(SPLIT_NAMES)
    for split in SPLIT_NAMES:
        # 3 条 fake 行:1 条保留、1 条 31s 音频过滤、1 条 label 长度 9 > 5 过滤
        assert manifest["splits"][split] == {"rows_before": 3, "rows_after": 1}
    assert manifest["dtype"] == "float16"
    assert manifest["subset_head"] is None
    assert manifest["feature_dir"] == str(data_root / "full" / "faketest")

    on_disk = json.loads((data_root / "full" / "faketest" / "manifest.json").read_text())
    assert on_disk == manifest

    ds = load_from_disk(str(data_root / "full" / "faketest" / "train.clean.100"))
    assert len(ds) == 1
    assert "float16" in str(ds.features["input_features"])
    assert ds[0]["length"] == 16000
    assert ds[0]["labels"] == [ord("o") % 32, ord("k") % 32]


def test_prepare_respects_subset_head(data_root):
    manifest = full_data.prepare_full_dataset(CFG, FakeAdapter, subset_head=1)
    for split in SPLIT_NAMES:
        assert manifest["splits"][split] == {"rows_before": 1, "rows_after": 1}
    assert manifest["subset_head"] == 1


def test_params_hash_covers_feature_params_only():
    base = full_data.params_hash(CFG)

    irrelevant = copy.deepcopy(CFG)
    irrelevant["training"]["learning_rate"] = 9.9e-9
    irrelevant["data"]["n_train"] = 7
    irrelevant["run_name"] = "other_run"
    irrelevant["smoke"] = {"overfit1_steps": 1}
    assert full_data.params_hash(irrelevant) == base

    model_changed = copy.deepcopy(CFG)
    model_changed["model_size"] = "not-fake"
    assert full_data.params_hash(model_changed) != base

    filter_changed = copy.deepcopy(CFG)
    filter_changed["data"]["max_label_len"] = 448
    assert full_data.params_hash(filter_changed) != base


CFG_WITH_MODEL_SECTION = {
    "run_name": "full_data_unit_model_section",
    "model": {
        "size": "medium",
        "gradient_checkpointing": True,
        "generation_max_length": 225,
        "apply_spec_augment": False,
    },
    "training": {"learning_rate": 1.0e-4},
    "data": {"max_label_len": 448, "max_audio_seconds": 30.0},
}


def test_params_hash_excludes_non_feature_model_keys():
    """model.gradient_checkpointing / generation_max_length / apply_spec_augment 是训练/解码期
    旋钮(见 asrfs/whisper/config.yaml),不改变任何一个特征字节,不应参与 hash;
    model.size 与 data.max_label_len 才是真正影响特征的 key,必须参与 hash。
    """
    base = full_data.params_hash(CFG_WITH_MODEL_SECTION)

    gc_changed = copy.deepcopy(CFG_WITH_MODEL_SECTION)
    gc_changed["model"]["gradient_checkpointing"] = False
    assert full_data.params_hash(gc_changed) == base

    genlen_changed = copy.deepcopy(CFG_WITH_MODEL_SECTION)
    genlen_changed["model"]["generation_max_length"] = 448
    assert full_data.params_hash(genlen_changed) == base

    spec_changed = copy.deepcopy(CFG_WITH_MODEL_SECTION)
    spec_changed["model"]["apply_spec_augment"] = True
    assert full_data.params_hash(spec_changed) == base

    size_changed = copy.deepcopy(CFG_WITH_MODEL_SECTION)
    size_changed["model"]["size"] = "large"
    assert full_data.params_hash(size_changed) != base

    label_len_changed = copy.deepcopy(CFG_WITH_MODEL_SECTION)
    label_len_changed["data"]["max_label_len"] = 5
    assert full_data.params_hash(label_len_changed) != base


def test_load_full_dataset_roundtrip(data_root):
    full_data.prepare_full_dataset(CFG, FakeAdapter)
    train, eval_ds = full_data.load_full_dataset(CFG, model_name="faketest")
    assert len(train) == 3  # 三个 train split 各存活 1 条,concatenate 后 3 条
    assert len(eval_ds) == 1
    assert set(train.column_names) >= {"input_features", "labels", "length"}


def test_prepare_full_dataset_removes_stale_manifest_on_crash(monkeypatch, data_root):
    """re-run 若中途崩溃(比如换了 cfg 后第二个 split 失败),旧 manifest 必须已经不在了,
    否则 load_full_dataset(旧 cfg) 会悄悄拼出一份新旧 split 混杂的 train set。
    """
    full_data.prepare_full_dataset(CFG, FakeAdapter)
    manifest_path = data_root / "full" / "faketest" / "manifest.json"
    assert manifest_path.is_file()

    other_cfg = copy.deepcopy(CFG)
    other_cfg["model_size"] = "not-fake"  # 不同 params_hash

    calls = {"n": 0}

    def _crash_on_second_split(config, split, subset_head=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom: simulated crash mid re-run")
        yield from _fake_rows(f"{config}.{split}")

    monkeypatch.setattr(full_data, "_stream_split", _crash_on_second_split)

    with pytest.raises(Exception):
        full_data.prepare_full_dataset(other_cfg, FakeAdapter)

    assert not manifest_path.exists()
    with pytest.raises(FileNotFoundError):
        full_data.load_full_dataset(CFG, model_name="faketest")


def test_load_full_dataset_guards(data_root):
    with pytest.raises(ValueError):
        full_data.load_full_dataset(CFG)  # 缺 model_name
    with pytest.raises(FileNotFoundError):
        full_data.load_full_dataset(CFG, model_name="nosuch")

    full_data.prepare_full_dataset(CFG, FakeAdapter)
    stale = copy.deepcopy(CFG)
    stale["model_size"] = "not-fake"
    with pytest.raises(ValueError):
        full_data.load_full_dataset(stale, model_name="faketest")


ADAPTER_CASES = [
    ("asrfs.whisper", "whisper"),
    ("asrfs.parakeet", "parakeet"),
]


@pytest.mark.parametrize("pkg_name,expected_model", ADAPTER_CASES)
def test_build_dataset_full_delegates_to_load_full_dataset(monkeypatch, pkg_name, expected_model):
    pkg = importlib.import_module(pkg_name)
    cfg = yaml.safe_load(Path(f"asrfs/{expected_model}/config.yaml").read_text())

    seen = {}

    def fake_load(cfg_in, model_name=None):
        seen["model_name"] = model_name
        return ("train-sentinel", "eval-sentinel")

    monkeypatch.setattr(full_data, "load_full_dataset", fake_load)
    out = pkg.build_dataset(cfg, None, mode="full")

    assert out == ("train-sentinel", "eval-sentinel")
    assert seen["model_name"] == expected_model


def test_params_hash_speed_perturb_missing_equals_default():
    base = full_data.params_hash(CFG)
    explicit = {**CFG, "data": {**(CFG.get("data") or {}), "speed_perturb": [1.0]}}
    assert full_data.params_hash(explicit) == base


def test_params_hash_speed_perturb_changes_hash():
    base = full_data.params_hash(CFG)
    sp = {**CFG, "data": {**(CFG.get("data") or {}), "speed_perturb": [0.9, 1.0, 1.1]}}
    assert full_data.params_hash(sp) != base


def test_params_hash_ignores_augment_section():
    base = full_data.params_hash(CFG)
    aug = {**CFG, "augment": {"spec_augment": {"time_axis": 0, "p": 0.9}}}
    assert full_data.params_hash(aug) == base


def test_prepare_speed_perturb_triples_train_rows(data_root):
    # FakeAdapter 每 split 1 行原始音频;speed_perturb 3 态 → train split rows_after≈3、eval=1
    cfg = {**CFG, "data": {**(CFG.get("data") or {}), "speed_perturb": [0.9, 1.0, 1.1]}}
    manifest = full_data.prepare_full_dataset(cfg, FakeAdapter, subset_head=1)
    splits = manifest["splits"]
    # TRAIN_SPLIT_NAMES 每个 rows_after 应为 3(1 原始 ×3 factor,均通过 FakeAdapter 过滤)
    for name in full_data.TRAIN_SPLIT_NAMES:
        assert splits[name]["rows_after"] == 3, (name, splits[name])
    # eval split 不变速 → 1
    assert splits[full_data.EVAL_SPLIT_NAME]["rows_after"] == 1


def test_perturb_speed_length_relation():
    audio = np.linspace(-1, 1, 16000, dtype=np.float32)
    slow = full_data._perturb_speed(audio, 16000, 0.9)
    fast = full_data._perturb_speed(audio, 16000, 1.1)
    same = full_data._perturb_speed(audio, 16000, 1.0)
    assert len(slow) > 16000 > len(fast)          # 0.9 变慢变长,1.1 变快变短
    assert same is audio or np.array_equal(same, audio)  # 1.0 恒等


def test_params_hash_includes_tokenizer_fingerprint():
    # round-2 审计 F1/F2:tokenizer 身份必须参与 hash,否则重训 BPE/Hub 更新
    # 会静默复用旧 label 缓存。
    base = full_data.params_hash(CFG)
    assert base != full_data.params_hash(CFG, tokenizer_fingerprint="spm:aaaa")
    assert full_data.params_hash(CFG, tokenizer_fingerprint="spm:aaaa") != full_data.params_hash(
        CFG, tokenizer_fingerprint="spm:bbbb"
    )


def test_tokenizer_fingerprint_registry(tmp_path, monkeypatch):
    # 未注册(FakeAdapter 的 faketest)→ None,prepare/load 语义不变
    assert full_data._tokenizer_fingerprint("faketest", CFG) is None
    # 四个真模型都必须声明指纹
    fp_x = full_data._tokenizer_fingerprint("x_asr", {})
    assert fp_x and fp_x.startswith("spm:") and "@" in fp_x
    assert full_data._tokenizer_fingerprint("parakeet", {}) == (
        "nvidia/parakeet-ctc-0.6b@ad09ba1cc62743fbc9814de5d2016fca9096485a"
    )
    assert full_data._tokenizer_fingerprint("sensevoice", {}) == (
        "nvidia/parakeet-ctc-0.6b@ad09ba1cc62743fbc9814de5d2016fca9096485a"
    )
    fp_w = full_data._tokenizer_fingerprint("whisper", {"model": {"size": "medium"}})
    assert fp_w == "openai/whisper-medium.en@2e98eb6279edf5095af0c8dedb36bdec0acd172b"
    # x_asr 指纹跟随 spm 文件内容:换文件字节 → 指纹变 → hash 变 → 旧特征判 stale
    from asrfs.x_asr import model as xm

    fake = tmp_path / "other.model"
    fake.write_bytes(b"retrained-bpe-bytes")
    monkeypatch.setattr(xm, "_BPE_MODEL", fake)
    fp_retrained = full_data._tokenizer_fingerprint("x_asr", {})
    assert fp_retrained != fp_x
    assert full_data.params_hash(CFG, tokenizer_fingerprint=fp_x) != full_data.params_hash(
        CFG, tokenizer_fingerprint=fp_retrained
    )


def test_duration_filter_is_post_perturb(data_root, monkeypatch):
    # 时长过滤必须作用在【变速后】音频:28s 原始行在 0.9(变长 →31s)应超 30s 上限被弃,
    # 1.0(28s)/1.1(25s)保留 → train rows_after==2。若误在原始长度上过滤(退化为
    # len(row["audio_array"])),三态都 <30s 全保留 →3,本断言即挂,正好锁死该回归。
    sr = 16000

    def _one_28s_row(config, split, subset_head=None):
        yield {"id": f"{config}.{split}-28s", "audio_array": np.zeros(28 * sr, dtype=np.float32),
               "sampling_rate": sr, "text": "ok"}

    monkeypatch.setattr(full_data, "_stream_split", _one_28s_row)
    cfg = {**CFG, "data": {**(CFG.get("data") or {}), "speed_perturb": [0.9, 1.0, 1.1]}}
    manifest = full_data.prepare_full_dataset(cfg, FakeAdapter, subset_head=1)
    splits = manifest["splits"]
    for name in full_data.TRAIN_SPLIT_NAMES:
        assert splits[name]["rows_after"] == 2, (name, splits[name])  # 0.9 弃、1.0/1.1 留
    assert splits[full_data.EVAL_SPLIT_NAME]["rows_after"] == 1        # eval 不变速,28s 保留


def test_length_column_is_perturbed_sample_count(data_root, monkeypatch):
    # length 列必须是【变速后】采样点数(供 group_by_length 分桶);单条 1s 行 ×[0.9,1.0,1.1]
    # 应产三种不同 length(0.9>16000>1.1),1.0 恰 16000。误设常量或原始长度 → 三值相等,挂。
    sr = 16000

    def _one_1s_row(config, split, subset_head=None):
        yield {"id": f"{config}.{split}-1s", "audio_array": np.zeros(sr, dtype=np.float32),
               "sampling_rate": sr, "text": "ok"}

    monkeypatch.setattr(full_data, "_stream_split", _one_1s_row)
    cfg = {**CFG, "data": {**(CFG.get("data") or {}), "speed_perturb": [0.9, 1.0, 1.1]}}
    full_data.prepare_full_dataset(cfg, FakeAdapter, subset_head=1)
    ds = load_from_disk(str(data_root / "full" / "faketest" / "train.clean.100"))
    lengths = sorted(int(x) for x in ds["length"])
    assert len(lengths) == 3 and len(set(lengths)) == 3, lengths  # 三态各异
    assert lengths[0] < sr < lengths[2]                            # 1.1 最短 < 16000 < 0.9 最长
    assert sr in lengths                                           # 1.0 恰为原始 16000


def test_load_full_dataset_rejects_subset_features(data_root, monkeypatch):
    monkeypatch.delenv("ASRFS_ALLOW_SUBSET", raising=False)
    full_data.prepare_full_dataset(CFG, FakeAdapter, subset_head=2)  # 写 subset manifest
    with pytest.raises(RuntimeError, match="subset_head"):
        full_data.load_full_dataset(CFG, model_name="faketest")


def test_load_full_dataset_subset_allowed_via_env(data_root, monkeypatch):
    monkeypatch.setenv("ASRFS_ALLOW_SUBSET", "1")
    full_data.prepare_full_dataset(CFG, FakeAdapter, subset_head=2)
    train, eval_ds = full_data.load_full_dataset(CFG, model_name="faketest")
    assert len(train) > 0 and len(eval_ds) > 0


def test_load_full_dataset_augments_train_only(data_root, monkeypatch):
    # 门控双向验证走结构断言(spec §5):FakeAdapter 特征是常值,mean 填充对常值
    # 是 no-op,值变异断言不可靠;行为正确性由 test_augment.py 的种子化单测负责。
    # 本测试用 subset_head 预备特征做子集验证,须显式放行新的 subset 守卫(F3)。
    monkeypatch.setenv("ASRFS_ALLOW_SUBSET", "1")
    full_data.prepare_full_dataset(CFG, FakeAdapter, subset_head=2)
    plain_train, plain_eval = full_data.load_full_dataset(CFG, model_name="faketest")
    # 反向:无 augment 段 → 两个 split 都无 custom transform
    assert plain_train.format["type"] != "custom"
    assert plain_eval.format["type"] != "custom"

    aug_cfg = {
        **CFG,
        "augment": {"spec_augment": {"time_axis": 0, "p": 1.0}},
    }
    train, eval_ds = full_data.load_full_dataset(aug_cfg, model_name="faketest")
    # 正向:train 挂了 custom transform,eval 没挂
    assert train.format["type"] == "custom"
    assert eval_ds.format["type"] != "custom"
    # length 列访问不被 transform 破坏(group_by_length 路径,spec 待钉 5)
    assert list(train["length"]) == list(plain_train["length"])
