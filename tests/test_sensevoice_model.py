"""SenseVoice 模型结构/前向/损失(spec 2026-07-03 §4/§7)。tiny config 全 CPU。"""

import math

import pytest
import torch

from asrfs.sensevoice.model import (
    EXPECTED_FROZEN,
    LABEL_PAD_ID,
    LOSS_FAMILY,
    SenseVoiceConfig,
    SenseVoiceForCTC,
    apply_lfr,
    build_model,
    init_report,
)

TINY = dict(
    vocab_size=11, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=64, fsmn_kernel_size=11, sanm_shift=0, lfr_m=7, lfr_n=6,
    num_mel_bins=8, dropout=0.0, blank_id=10,
)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return SenseVoiceForCTC(SenseVoiceConfig(**TINY)).eval()


def test_constants():
    assert LOSS_FAMILY == "ctc"
    assert LABEL_PAD_ID == 1024
    assert EXPECTED_FROZEN == set()


def test_apply_lfr_shape_and_content():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
    out = apply_lfr(x, lfr_m=7, lfr_n=6)
    assert out.shape == (1, math.ceil(12 / 6), 2 * 7)
    # 窗 0 覆盖 [左pad 3 帧(重复首帧), 帧 0..3]:前 3 段 = 首帧
    assert torch.equal(out[0, 0, :2], x[0, 0])
    assert torch.equal(out[0, 0, 2:4], x[0, 0])
    assert torch.equal(out[0, 0, 6:8], x[0, 0])
    assert torch.equal(out[0, 0, 8:10], x[0, 1])
    # 窗 1 起点 = 帧 6-3=3
    assert torch.equal(out[0, 1, :2], x[0, 3])


def test_forward_shapes_and_lengths(tiny_model):
    b, t, d_in = 2, 36, TINY["num_mel_bins"]
    feats = torch.randn(b, t, d_in)
    mask = torch.ones(b, t)
    mask[1, 24:] = 0  # 样本 2 真实长 24
    out_masked = tiny_model(input_features=feats, attention_mask=mask)
    assert out_masked.logits.shape == (b, math.ceil(t / 6), TINY["vocab_size"])
    # mask 必须有实际效果:样本 2 去掉 mask(视为全 36 帧有效)后 logits 应变化
    out_unmasked = tiny_model(input_features=feats, attention_mask=torch.ones(b, t))
    assert not torch.allclose(out_masked.logits[1], out_unmasked.logits[1])


def test_apply_lfr_non_multiple_length():
    # T=25 不是 lfr_n=6 的倍数:T'=ceil(25/6)=5(floor 会给 4,mutation killer)。
    x = torch.randn(1, 25, 2)
    out = apply_lfr(x, 7, 6)
    assert out.shape == (1, 5, 14)


def test_forward_non_multiple_length_and_mask(tiny_model):
    # T=25 非 lfr_n 倍数:输出帧数须为 ceil(25/6)=5。
    feats25 = torch.randn(1, 25, TINY["num_mel_bins"])
    out25 = tiny_model(input_features=feats25, attention_mask=torch.ones(1, 25))
    assert out25.logits.shape[1] == 5

    # mask 语义:batch 内样本 0 真实长 13(pad 到 36,pad 区显式补零),
    # 应与单独跑 13 帧的前 ceil(13/6)=3 个 LFR 窗完全一致(两侧在有效音频外
    # 都是零延展,FSMN/attention 的 mask 保证等价)。
    torch.manual_seed(2)
    feats13 = torch.randn(1, 13, TINY["num_mel_bins"])
    padded0 = torch.cat([feats13, torch.zeros(1, 23, TINY["num_mel_bins"])], dim=1)
    other = torch.randn(1, 36, TINY["num_mel_bins"])
    batch_feats = torch.cat([padded0, other], dim=0)
    mask = torch.ones(2, 36)
    mask[0, 13:] = 0
    out_batch = tiny_model(input_features=batch_feats, attention_mask=mask)
    out_single = tiny_model(input_features=feats13, attention_mask=torch.ones(1, 13))
    assert out_single.logits.shape[1] == 3
    assert torch.allclose(out_batch.logits[0, :3], out_single.logits[0], atol=1e-5)


def test_ctc_input_lengths_use_ceil():
    # sharp mutation killer:T=25 → T'=ceil(25/6)=5。5 个非 blank 标签,
    # T'=5>=L=5 时 CTC 可行、loss 有限且 >0;若 out_lengths 误用 floor(=4),
    # T'<L 不可行,zero_infinity 会把 loss 钳到恰好 0。
    torch.manual_seed(0)
    model = SenseVoiceForCTC(SenseVoiceConfig(**TINY)).train()
    feats = torch.randn(1, 25, TINY["num_mel_bins"])
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    out = model(input_features=feats, attention_mask=torch.ones(1, 25), labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.loss.item() > 0


def test_pad_invariance(tiny_model):
    # 真实长度 24(6 的倍数,全部 LFR 窗落在有效+左 pad 区,spec §7)
    torch.manual_seed(1)
    feats = torch.randn(1, 24, TINY["num_mel_bins"])
    short = tiny_model(input_features=feats, attention_mask=torch.ones(1, 24)).logits
    padded_feats = torch.cat([feats, torch.zeros(1, 12, TINY["num_mel_bins"])], dim=1)
    mask = torch.cat([torch.ones(1, 24), torch.zeros(1, 12)], dim=1)
    long = tiny_model(input_features=padded_feats, attention_mask=mask).logits
    assert torch.allclose(short, long[:, : short.size(1)], atol=1e-5)


def test_loss_finite_and_backprop():
    torch.manual_seed(0)
    model = SenseVoiceForCTC(SenseVoiceConfig(**TINY)).train()
    feats = torch.randn(2, 36, TINY["num_mel_bins"])
    labels = torch.full((2, 4), TINY["blank_id"], dtype=torch.long)
    labels[0, :3] = torch.tensor([1, 2, 3])
    labels[1, :2] = torch.tensor([4, 5])
    out = model(input_features=feats, attention_mask=torch.ones(2, 36), labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert sum(g > 0 for g in grads) > 0


def test_all_pad_labels_do_not_crash():
    torch.manual_seed(0)
    model = SenseVoiceForCTC(SenseVoiceConfig(**TINY)).train()
    feats = torch.randn(1, 12, TINY["num_mel_bins"])
    labels = torch.full((1, 4), TINY["blank_id"], dtype=torch.long)
    out = model(input_features=feats, attention_mask=torch.ones(1, 12), labels=labels)
    assert out.loss is not None  # target_lengths=0,zero_infinity 兜底,不许 crash


def test_default_build_model_structure_and_init():
    model = build_model({})
    rep = init_report(model)
    assert 25_000_000 < rep["params_total"] < 35_000_000
    assert rep["params_trainable"] == rep["params_total"]
    assert rep["frozen"] == set()
    assert abs(rep["sample_std"] - 0.02) < 0.005
    assert len(model.encoders0) == 1 and len(model.encoders) == 15
    fsmn = model.encoders[0].self_attn.fsmn_block
    assert fsmn.groups == 384 and fsmn.kernel_size == (11,) and fsmn.bias is None
    # 首层投影:in 560 = 80*7
    assert model.encoders0[0].self_attn.linear_q_k_v.in_features == 560


def test_default_config_loss_forward():
    # 默认配置(vocab_size=1025, blank_id=1024)下真实跑一次 loss:
    # 守住 vocab/blank 一致性从未被真实 forward 验证过的缺口。
    torch.manual_seed(0)
    model = build_model({}).train()
    feats = torch.randn(1, 25, 80)
    labels = torch.tensor([[5, 17, 900]])
    out = model(input_features=feats, attention_mask=torch.ones(1, 25), labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.loss.item() > 0


def test_model_config_overrides():
    model = build_model({"model": {"num_hidden_layers": 3, "hidden_size": 64,
                                   "num_attention_heads": 2, "intermediate_size": 128}})
    assert len(model.encoders) == 2
    assert model.config.hidden_size == 64
