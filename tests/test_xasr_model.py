"""X-ASR 模型:结构/长度公式(非整除长度)/RNN-T loss/greedy/往返。tiny 全 CPU。"""

import pytest
import torch

from asrfs.x_asr.model import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    XASRConfig,
    XASRForRNNT,
    build_model,
    init_report,
)

TINY = dict(
    vocab_size=11,
    blank_id=10,
    num_mel_bins=80,
    downsampling_factor=(1, 2),
    num_encoder_layers=(1, 1),
    encoder_dim=(24, 32),
    encoder_unmasked_dim=(24, 24),
    query_head_dim=8,
    pos_head_dim=4,
    value_head_dim=8,
    pos_dim=16,
    num_heads=(2, 2),
    feedforward_dim=(48, 48),
    cnn_module_kernel=(7, 7),
    decoder_dim=16,
    joiner_dim=16,
    context_size=2,
    dropout=0.0,
)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return XASRForRNNT(XASRConfig(**TINY)).eval()


def test_constants():
    assert LOSS_FAMILY == "rnnt"
    assert LABEL_PAD_ID == 1024
    assert EXPECTED_FROZEN == set()


@pytest.mark.parametrize("t_in", [25, 30, 31, 36])
def test_encode_length_formula_non_multiples(tiny_model, t_in):
    feats = torch.randn(1, t_in, 80)
    enc, lens = tiny_model.encode(feats, torch.ones(1, t_in))
    expect = ((t_in - 7) // 2 + 1) // 2
    assert lens.item() == expect
    assert enc.shape[1] >= expect  # batch pad may exceed
    assert enc.shape[2] == 32  # batch-major, C=max(encoder_dim)


def test_encode_batch_mask(tiny_model):
    # 两条不同真实长度(都非整除),短样本有效段与单独前向一致
    torch.manual_seed(1)
    a = torch.randn(1, 25, 80)
    b = torch.randn(1, 37, 80)
    batch = torch.zeros(2, 37, 80)
    batch[0, :25] = a[0]
    batch[1] = b[0]
    mask = torch.zeros(2, 37)
    mask[0, :25] = 1
    mask[1] = 1
    enc_b, lens_b = tiny_model.encode(batch, mask)
    enc_a, lens_a = tiny_model.encode(a, torch.ones(1, 25))
    la = lens_a.item()
    assert lens_b[0].item() == la
    # icefall zipformer 的最后一个输出帧不是 batch-不变的:末端 SimpleDownsample
    # (_vendor/zipformer.py SimpleDownsample.forward)不 mask padding,单样本对奇数
    # 长度用"重复末元素"补齐,而 batch 里该窗口含真实(已处理的)padding 帧 → 末帧
    # 必然不同(实测 ~0.3)。有效区内部帧对齐到 ~4e-4(batch-vs-单样本 BLAS 浮点噪声
    # + 未 mask 的前端 Conv2dSubsampling)。故比对内部帧(排除末帧),容差 1e-3。
    # 这是 vendored 编码器的固有性质,忠实复刻自 icefall forward_encoder(见接口文档 §f)。
    assert torch.allclose(enc_b[0, : la - 1], enc_a[0, : la - 1], atol=1e-3)


def test_forward_loss_finite_and_backprop():
    torch.manual_seed(0)
    model = XASRForRNNT(XASRConfig(**TINY)).train()
    feats = torch.randn(2, 41, 80)
    labels = torch.full((2, 5), TINY["blank_id"], dtype=torch.long)
    labels[0, :3] = torch.tensor([1, 2, 3])
    labels[1, :4] = torch.tensor([4, 5, 6, 7])
    out = model(input_features=feats, attention_mask=torch.ones(2, 41), labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_forward_no_labels_returns_encoder_out(tiny_model):
    feats = torch.randn(1, 25, 80)
    out = tiny_model(input_features=feats, attention_mask=torch.ones(1, 25))
    assert out.loss is None
    assert out.encoder_out is not None and out.encoder_out_lens is not None
    assert not hasattr(out, "logits") or out.logits is None  # spec:不返回 joint


def test_greedy_decode_returns_token_lists(tiny_model):
    feats = torch.randn(2, 41, 80)
    mask = torch.ones(2, 41)
    mask[1, 25:] = 0
    hyps = tiny_model.greedy_decode(feats, mask)
    assert isinstance(hyps, list) and len(hyps) == 2
    for h in hyps:
        assert isinstance(h, list)
        assert all(isinstance(t, int) and 0 <= t < TINY["vocab_size"] and t != TINY["blank_id"] for t in h)


def test_checkpoint_roundtrip(tmp_path, tiny_model):
    out_dir = tmp_path / "ckpt"
    tiny_model.save_pretrained(out_dir)
    reloaded = XASRForRNNT.from_pretrained(out_dir).eval()
    feats = torch.randn(1, 31, 80)
    a, la = tiny_model.encode(feats, torch.ones(1, 31))
    b, lb = reloaded.encode(feats, torch.ones(1, 31))
    assert torch.allclose(a, b, atol=1e-6) and la.item() == lb.item()


def test_default_build_model_structure_and_init():
    model = build_model({})
    rep = init_report(model)
    assert 18_000_000 < rep["params_total"] < 30_000_000
    assert rep["params_trainable"] == rep["params_total"]
    assert rep["frozen"] == set()
    # icefall 初始化非 0.02:_init_weights 必须是 no-op(押 fast-init 不重随机化由 roundtrip 测试管)
    assert model.config.vocab_size == 1025 and model.config.blank_id == 1024
    assert model.decoder.blank_id == 1024 and model.decoder.context_size == 2
    assert model.joiner.output_linear.out_features == 1025


def test_default_config_loss_forward():
    torch.manual_seed(0)
    model = build_model({}).train()
    feats = torch.randn(1, 41, 80)
    labels = torch.tensor([[5, 17, 900]])
    out = model(input_features=feats, attention_mask=torch.ones(1, 41), labels=labels)
    assert torch.isfinite(out.loss) and out.loss > 0
