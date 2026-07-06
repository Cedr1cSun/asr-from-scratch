"""CTC 族共享件(asrfs/common/ctc.py)与 parakeet 薄 re-export 的同一性。"""

import numpy as np
import torch


def test_common_ctc_importable():
    from asrfs.common.ctc import (  # noqa: F401
        OVERFIT1_REPLICAS,
        CTCCollator,
        CTCProcessorBundle,
        build_ctc_dataset,
        build_ctc_trainer,
        ctc_decode,
        ctc_greedy_decode,
        prepare_ctc_example,
    )

    assert OVERFIT1_REPLICAS == 100


def test_parakeet_reexports_are_same_objects():
    import asrfs.common.ctc as ctc
    import asrfs.parakeet.dataset as pds

    assert pds.ParakeetCollator is ctc.CTCCollator
    assert pds.ctc_greedy_decode is ctc.ctc_greedy_decode
    assert pds.OVERFIT1_REPLICAS == ctc.OVERFIT1_REPLICAS


def test_collator_upcasts_float16_input_features_to_float32():
    """full 模式磁盘特征缓存是 float16(asrfs/common/full_data.py FEATURE_DTYPE);
    --precision fp32 训练路径没有 autocast,pad 后的 float16 张量喂进模型第一层
    conv/layer_norm 会跟 float32 bias 类型不匹配报错。collator 必须把 padded
    input_features 转回 float32(对 smoke 路径本就是 float32 的特征,.float() 是
    恒等操作,无副作用)。"""
    from asrfs.common.ctc import CTCCollator, prepare_ctc_example
    from asrfs.parakeet.model import build_feature_extractor, build_tokenizer

    fe, tok = build_feature_extractor(), build_tokenizer()
    exs = [
        prepare_ctc_example(
            {"audio_array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000, "text": "hi"},
            fe,
            tok,
        ),
        prepare_ctc_example(
            {
                "audio_array": np.zeros(32000, dtype=np.float32),
                "sampling_rate": 16000,
                "text": "hello there",
            },
            fe,
            tok,
        ),
    ]
    for ex in exs:
        # 模拟磁盘缓存:预计算特征以 float16 落盘(见 full_data.py _prepared_rows)
        ex["input_features"] = np.asarray(ex["input_features"], dtype=np.float16)

    collator = CTCCollator(fe, pad_label_id=tok.vocab_size)
    batch = collator(exs)

    assert batch["input_features"].dtype == torch.float32
