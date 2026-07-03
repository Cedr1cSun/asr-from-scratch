"""vendored icefall 件:import 干净(无 k2/lhotse)、小模型可前向、长度公式吻合。"""

import subprocess
import sys

import pytest
import torch

VENDOR = "asrfs.x_asr._vendor"


def test_vendor_imports_without_k2_or_lhotse():
    code = (
        "import sys;"
        "sys.modules['k2']=None; sys.modules['lhotse']=None;"  # 存在即炸的哨兵
        f"import {VENDOR}.zipformer, {VENDOR}.scaling, {VENDOR}.subsampling,"
        f" {VENDOR}.decoder, {VENDOR}.joiner, {VENDOR}.optim, {VENDOR}.icefall_compat;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_vendor_no_k2_lhotse_references():
    import pathlib

    vendor_dir = pathlib.Path("asrfs/x_asr/_vendor")
    for f in vendor_dir.glob("*.py"):
        src = f.read_text()
        assert "import k2" not in src, f"{f} still imports k2"
        assert "from lhotse" not in src, f"{f} still imports lhotse"


@pytest.fixture(scope="module")
def tiny_encoder():
    from asrfs.x_asr._vendor.subsampling import Conv2dSubsampling
    from asrfs.x_asr._vendor.zipformer import Zipformer2

    torch.manual_seed(0)
    embed = Conv2dSubsampling(in_channels=80, out_channels=24, dropout=0.0)
    enc = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2),
        num_encoder_layers=(1, 1),
        encoder_dim=(24, 32),
        encoder_unmasked_dim=(24, 24),
        query_head_dim=(8, 8),
        pos_head_dim=(4, 4),
        value_head_dim=(8, 8),
        pos_dim=16,
        num_heads=(2, 2),
        feedforward_dim=(48, 48),
        cnn_module_kernel=(7, 7),
        dropout=0.0,
        warmup_batches=4000.0,
        causal=False,
        chunk_size=(-1,),
        left_context_frames=(-1,),
    )
    return embed, enc


@pytest.mark.parametrize("t_in", [25, 30, 31, 36])  # 非整除长度必须在(全局约束)
def test_subsample_and_encoder_length_formula(tiny_encoder, t_in):
    from asrfs.x_asr._vendor.icefall_compat import make_pad_mask

    embed, enc = tiny_encoder
    embed.eval(), enc.eval()
    x = torch.randn(1, t_in, 80)
    lens = torch.tensor([t_in])
    with torch.no_grad():
        x1, l1 = embed(x, lens)
        assert l1.item() == (t_in - 7) // 2
        assert x1.shape[1] == l1.item()
        mask = make_pad_mask(l1)
        out, l2 = enc(x1.permute(1, 0, 2), l1, mask)
    assert l2.item() == (l1.item() + 1) // 2
    assert out.shape[0] == out.shape[0] and out.shape[2] == 32  # max(encoder_dim)


def test_scaled_adam_and_eden_construct(tiny_encoder):
    from asrfs.x_asr._vendor.icefall_compat import get_parameter_groups_with_lrs
    from asrfs.x_asr._vendor.optim import Eden, ScaledAdam

    _, enc = tiny_encoder
    opt = ScaledAdam(
        get_parameter_groups_with_lrs(enc, lr=0.045, include_names=True),
        lr=0.045,
        clipping_scale=2.0,
    )
    sched = Eden(opt, lr_batches=7500, lr_epochs=3.5, warmup_start=0.1)
    assert sched is not None


def test_make_pad_mask_semantics():
    from asrfs.x_asr._vendor.icefall_compat import make_pad_mask

    m = make_pad_mask(torch.tensor([1, 3]))
    assert m.tolist() == [[False, True, True], [False, False, False]]
