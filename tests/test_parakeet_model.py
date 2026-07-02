from parakeet.model import build_model, build_tokenizer, init_report

def test_small_random_init():
    model = build_model()
    report = init_report(model)
    assert 20e6 < report["params_total"] < 40e6
    assert report["params_total"] == report["params_trainable"] or report["frozen"], "frozen set must be explicit"
    assert model.config.pad_token_id == model.config.vocab_size - 1

def test_tokenizer_vocab_alignment():
    tok = build_tokenizer()
    model = build_model()
    assert model.config.vocab_size == tok.vocab_size + 1
