"""适配契约表面锁定(spec §2:9 函数 + 3 常量,签名逐字冻结)。

harness 侧 check_contract.py 是运行时守门;本测试把同一契约钉进 golden 仓
回归套件——契约漂移在本仓 pytest 即红,不用等 harness 管线才发现。
"""

import importlib
import inspect

import pytest

PACKAGES = ["asrfs.whisper", "asrfs.parakeet", "asrfs.sensevoice"]

CONTRACT_FUNCTIONS = {
    "build_processor": ["cfg"],
    "build_model": ["cfg"],
    "build_dataset": ["cfg", "processor", "mode"],
    "make_example": ["processor", "audio", "sampling_rate", "text"],
    "build_collator": ["cfg", "processor", "model"],
    "build_trainer": ["cfg", "model", "processor", "train_ds", "eval_ds", "collator", "overrides"],
    "decode": ["model", "processor", "batch"],
    "save_checkpoint": ["model", "processor", "out_dir"],
    "load_checkpoint": ["cfg", "ckpt_dir"],
}


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_nine_functions_present_with_pinned_signatures(pkg_name):
    pkg = importlib.import_module(pkg_name)
    for fn_name, required in CONTRACT_FUNCTIONS.items():
        fn = getattr(pkg, fn_name, None)
        assert callable(fn), f"{pkg_name}.{fn_name} missing or not callable"
        params = list(inspect.signature(fn).parameters.values())
        head = [p.name for p in params[: len(required)]]
        assert head == required, f"{pkg_name}.{fn_name}: params {head} != pinned {required}"
        for extra in params[len(required):]:
            assert extra.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ) or extra.default is not inspect.Parameter.empty, (
                f"{pkg_name}.{fn_name}: extra param {extra.name} must have a default"
            )


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_three_constants_present_and_typed(pkg_name):
    pkg = importlib.import_module(pkg_name)
    assert isinstance(pkg.LOSS_FAMILY, str) and pkg.LOSS_FAMILY in {"ce", "ctc"}
    assert isinstance(pkg.LABEL_PAD_ID, int) and not isinstance(pkg.LABEL_PAD_ID, bool)
    assert isinstance(pkg.EXPECTED_FROZEN, set)


def test_pinned_per_model_values():
    whisper = importlib.import_module("asrfs.whisper")
    parakeet = importlib.import_module("asrfs.parakeet")

    assert whisper.LOSS_FAMILY == "ce"
    assert whisper.LABEL_PAD_ID == -100
    assert whisper.EXPECTED_FROZEN == {"model.encoder.embed_positions.weight"}

    assert parakeet.LOSS_FAMILY == "ctc"
    assert parakeet.LABEL_PAD_ID == 1024  # blank = ParakeetTokenizerFast.vocab_size
    assert parakeet.EXPECTED_FROZEN == set()

    sensevoice = importlib.import_module("asrfs.sensevoice")

    assert sensevoice.LOSS_FAMILY == "ctc"
    assert sensevoice.LABEL_PAD_ID == 1024  # blank = ParakeetTokenizerFast.vocab_size
    assert sensevoice.EXPECTED_FROZEN == set()
