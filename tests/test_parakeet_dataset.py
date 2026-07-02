import numpy as np
import torch

from parakeet.dataset import ParakeetCollator, ctc_greedy_decode, prepare_example
from parakeet.model import build_feature_extractor, build_tokenizer


def _fake_sample(seconds: float, text: str) -> dict:
    return {
        "audio_array": np.zeros(int(16000 * seconds), dtype=np.float32),
        "sampling_rate": 16000,
        "text": text,
    }


def test_prepare_and_collate_variable_length():
    fe, tok = build_feature_extractor(), build_tokenizer()
    exs = [
        prepare_example(_fake_sample(2.0, "HELLO WORLD"), fe, tok),
        prepare_example(_fake_sample(4.0, "A LONGER FAKE SENTENCE"), fe, tok),
    ]
    blank = tok.vocab_size
    batch = ParakeetCollator(fe, pad_label_id=blank)(exs)
    assert batch["input_features"].shape[0] == 2
    assert batch["attention_mask"].shape[0] == 2
    # 短样本的 mask 比长样本短
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()
    # labels 右侧 pad 到 blank
    lens = [len(ex["labels"]) for ex in exs]
    assert batch["labels"].shape[1] == max(lens)
    assert (batch["labels"][0, lens[0]:] == blank).all()


def test_ctc_greedy_decode_collapses():
    tok = build_tokenizer()
    blank = tok.vocab_size
    ids_hello = tok("hello", add_special_tokens=False)["input_ids"]
    a, b = ids_hello[0], ids_hello[-1]
    ids = torch.tensor([[a, a, blank, blank, b, b, blank]])
    out = ctc_greedy_decode(ids, tok, blank_id=blank)
    ref = tok.decode([a, b], skip_special_tokens=True)
    assert out == [ref]
