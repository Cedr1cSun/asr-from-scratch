from asrfs.whisper.model import SIZE_PRESETS, build_model, init_report

def test_presets_have_expected_dims():
    assert SIZE_PRESETS["small"]["d_model"] == 768
    assert SIZE_PRESETS["small"]["encoder_layers"] == 12
    assert SIZE_PRESETS["medium"]["d_model"] == 1024
    assert SIZE_PRESETS["medium"]["encoder_layers"] == 24

def test_medium_random_init():
    model = build_model("medium")
    report = init_report(model)
    assert 750e6 < report["params_total"] < 790e6
    assert report["frozen"] == {"model.encoder.embed_positions.weight"}
    assert 0.01 < report["enc_l0_q_std"] < 0.03
