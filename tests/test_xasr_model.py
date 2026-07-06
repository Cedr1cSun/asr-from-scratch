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
    assert LABEL_PAD_ID == 500
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


def test_loss_uses_per_sample_logit_lengths():
    """RNN-T loss 必须按每样本真实编码器帧数(encoder_out_lens)算,不是 batch 的
    padded 时间维 encoder_out.size(1)。两条不同真实长度合批,reduction="mean" 下
    batch loss 必须等于两条单样本 loss 的均值。用全长(full-length mutation)会把
    padding 帧塞进短样本网格,短样本 loss 被抬高 → 均值等式被破坏(实测 mutant
    batch≈24.7 vs 正确≈20.2)。45/27 是"干净"长度对:短样本末帧不受 batch 末端
    SimpleDownsample 的 padding 泄漏影响(见 test_encode_batch_mask),残差 ~1e-7。"""
    torch.manual_seed(0)
    model = XASRForRNNT(XASRConfig(**TINY)).eval()
    t_long, t_short = 45, 27
    torch.manual_seed(7)
    a = torch.randn(1, t_long, 80)
    b = torch.randn(1, t_short, 80)
    batch = torch.zeros(2, t_long, 80)
    batch[0] = a[0]
    batch[1, :t_short] = b[0]
    mask = torch.zeros(2, t_long)
    mask[0] = 1
    mask[1, :t_short] = 1
    blank = TINY["blank_id"]
    la_tok, lb_tok = [1, 2, 3, 4], [5, 6]
    labels = torch.full((2, 4), blank, dtype=torch.long)
    labels[0] = torch.tensor(la_tok)
    labels[1, :2] = torch.tensor(lb_tok)
    with torch.no_grad():
        batch_loss = model(input_features=batch, attention_mask=mask, labels=labels).loss
        loss_a = model(
            input_features=a, attention_mask=torch.ones(1, t_long),
            labels=torch.tensor([la_tok]),
        ).loss
        loss_b = model(
            input_features=b, attention_mask=torch.ones(1, t_short),
            labels=torch.tensor([lb_tok]),
        ).loss
    assert torch.allclose(batch_loss, (loss_a + loss_b) / 2, rtol=1e-4)


def test_forward_no_labels_returns_encoder_out(tiny_model):
    feats = torch.randn(1, 25, 80)
    out = tiny_model(input_features=feats, attention_mask=torch.ones(1, 25))
    assert out.loss is None
    assert out.encoder_out is not None and out.encoder_out_lens is not None
    assert not hasattr(out, "logits") or out.logits is None  # spec:不返回 joint


def test_greedy_decode_returns_token_lists(tiny_model):
    torch.manual_seed(0)
    feats = torch.randn(2, 41, 80)
    mask = torch.ones(2, 41)
    mask[1, 25:] = 0
    hyps = tiny_model.greedy_decode(feats, mask)
    assert isinstance(hyps, list) and len(hyps) == 2
    for h in hyps:
        assert isinstance(h, list)
        assert all(isinstance(t, int) and 0 <= t < TINY["vocab_size"] and t != TINY["blank_id"] for t in h)
    # emit-until-blank(msf 上限 32,见 greedy_decode)。tiny 未训练模型 argmax 近
    # 均匀、几乎不发 blank,每帧顶到安全帽 → token 数 == 帧数×cap,不再是旧 msf=1 的
    # 1-per-frame 精确计数。故弃 len==9/5 的硬计数,改断言结构不变量:批==单样本参照
    # (钉 per-sample lens 用法)、短样本 token 不多于长样本、都被 帧数×cap 硬上界。
    def enc(length):
        return ((length - 7) // 2 + 1) // 2

    cap = 32  # greedy_decode 的 max_sym_per_frame 默认值
    assert len(hyps[0]) <= enc(41) * cap  # 每帧至多 cap 次发射 → 帧数×cap 硬上界
    assert len(hyps[1]) <= enc(25) * cap
    assert 0 < len(hyps[1]) <= len(hyps[0])  # 短样本(帧更少)token 不多于长样本
    # 对每条样本单独跑 greedy 作参照:逐 token 相等 → per-sample lens 用法被钉死
    # (若误用 batch 长度/lens[0] 全长,短样本会多解到 9 帧,与单独 5 帧参照必不等)。
    ref0 = tiny_model.greedy_decode(feats[:1], torch.ones(1, 41))
    ref1 = tiny_model.greedy_decode(feats[1:2, :25], torch.ones(1, 25))
    assert hyps[0] == ref0[0]
    assert hyps[1] == ref1[0]


def test_greedy_decode_multi_emission_and_cap(tiny_model, monkeypatch):
    """标准 transducer greedy 的算法钉死(不需训练):脚本化 joiner.forward 直接喂
    argmax,验证(a)同帧 emit-until-blank 连发多 token 再前进,(b)一帧永不出 blank
    时恰好发 max_sym_per_frame 个后前进(安全帽生效)。杀 msf=1 死灰复燃的 mutant。"""
    blank, V = TINY["blank_id"], TINY["vocab_size"]

    def enc(length):
        return ((length - 7) // 2 + 1) // 2

    # (a) 2 帧输入(L=15 → enc=2)。按 joiner 调用序脚本:frame0 连发 [3,5] 再出 blank
    # 前进,frame1 首调用即 blank 前进。期望 emit [3,5],且恰 4 次 joiner 调用(钉前进)。
    assert enc(15) == 2
    script = [3, 5, blank]
    calls = {"n": 0}

    def scripted(encoder_out, decoder_out, project_input=True):
        i = calls["n"]
        calls["n"] += 1
        tok = script[i] if i < len(script) else blank
        logit = torch.full((1, 1, 1, V), -9.0)
        logit[0, 0, 0, tok] = 9.0
        return logit

    monkeypatch.setattr(tiny_model.joiner, "forward", scripted)
    hyps = tiny_model.greedy_decode(torch.randn(1, 15, 80), torch.ones(1, 15))
    assert hyps == [[3, 5]]
    assert calls["n"] == 4  # frame0: 3,5,blank(3 次) + frame1: blank(1 次)
    monkeypatch.undo()

    # (b) 1 帧输入(L=9 → enc=1),joiner 恒发非 blank(token 3)。cap=4 → 该帧恰发 4
    # 个后前进(无更多帧即结束);joiner 恰调 4 次(n_emit<cap 才调用,不会有第 5 次)。
    assert enc(9) == 1
    calls2 = {"n": 0}

    def always_nonblank(encoder_out, decoder_out, project_input=True):
        calls2["n"] += 1
        logit = torch.full((1, 1, 1, V), -9.0)
        logit[0, 0, 0, 3] = 9.0
        return logit

    monkeypatch.setattr(tiny_model.joiner, "forward", always_nonblank)
    hyps2 = tiny_model.greedy_decode(
        torch.randn(1, 9, 80), torch.ones(1, 9), max_sym_per_frame=4
    )
    assert hyps2 == [[3, 3, 3, 3]]
    assert calls2["n"] == 4  # 恰 cap 次,安全帽生效


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
    assert model.config.vocab_size == 501 and model.config.blank_id == 500
    assert model.decoder.blank_id == 500 and model.decoder.context_size == 2
    assert model.joiner.output_linear.out_features == 501


def test_default_config_loss_forward():
    torch.manual_seed(0)
    model = build_model({}).train()
    feats = torch.randn(1, 41, 80)
    labels = torch.tensor([[5, 17, 490]])
    out = model(input_features=feats, attention_mask=torch.ones(1, 41), labels=labels)
    assert torch.isfinite(out.loss) and out.loss > 0
