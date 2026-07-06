from pathlib import Path

import sentencepiece as spm

MODEL = Path(__file__).resolve().parent.parent / "asrfs" / "x_asr" / "bpe" / "librispeech_bpe500.model"


def test_bpe_model_exists_and_500():
    assert MODEL.is_file(), "run: python -m asrfs.x_asr.bpe.train_bpe"
    sp = spm.SentencePieceProcessor(model_file=str(MODEL))
    assert sp.get_piece_size() == 500


def test_bpe_roundtrip_lowercase():
    sp = spm.SentencePieceProcessor(model_file=str(MODEL))
    ids = sp.encode("the quick brown fox", out_type=int)
    assert len(ids) > 0 and all(isinstance(i, int) for i in ids)
    # decode 复原(spm 归一化空格,比对去空格小写)
    assert sp.decode(ids).lower().replace(" ", "") == "thequickbrownfox"


def test_spm_tokenizer_interface():
    from asrfs.x_asr.model import build_tokenizer

    tok = build_tokenizer()
    assert tok.vocab_size == 500
    ids = tok("the quick brown fox", add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    assert tok.decode(ids, skip_special_tokens=True).lower().replace(" ", "") == "thequickbrownfox"
