import sys
import tempfile
from pathlib import Path

from common.metrics import normalize_tokens, wer

SURE_ASR = "/home/cedric/WorkSpace-Exp/sure/src/sure_eval/evaluation/asr"


def test_normalize_matches_wenet_semantics():
    assert normalize_tokens("hello world") == ["HELLO", "WORLD"]
    assert normalize_tokens("a <noise> b") == ["A", "B"]  # tag stripped, empty dropped
    assert normalize_tokens("  spaced\tout ") == ["SPACED", "OUT"]


def test_wer_hand_computed():
    # ref 4 words, hyp has 1 substitution + 1 deletion => 2/4
    assert wer(["the cat sat down"], ["the bat sat"]) == 0.5
    assert wer(["a b"], ["a b"]) == 0.0


def test_wer_agrees_with_wenet_script():
    sys.path.insert(0, SURE_ASR)
    import wenet_compute_cer

    refs = ["the cat sat down", "hello world again", "one two three four five"]
    hyps = ["the bat sat", "hello word <noise> again", "one two three for five"]
    ours = wer(refs, hyps)

    with tempfile.TemporaryDirectory() as td:
        ref_file, hyp_file = Path(td) / "ref.txt", Path(td) / "hyp.txt"
        ref_file.write_text("".join(f"utt{i} {r}\n" for i, r in enumerate(refs)))
        hyp_file.write_text("".join(f"utt{i} {h}\n" for i, h in enumerate(hyps)))
        overall = wenet_compute_cer.compute_wer(str(ref_file), str(hyp_file))

    theirs = (overall["ins"] + overall["sub"] + overall["del"]) / overall["all"]
    assert abs(ours - theirs) < 1e-9
