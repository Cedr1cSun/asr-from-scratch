"""WER with text normalization identical to SURE-EVAL's wenet_compute_cer.py.

English WER path in that script: whitespace-split -> upper() -> strip <...>
tags -> drop empty tokens. Punctuation is NOT removed on this path (that
only happens in the Chinese CER characterize() path), so we don't either.
"""

import jiwer


def _stripoff_tags(x: str) -> str:
    # verbatim port of wenet_compute_cer.stripoff_tags
    if not x:
        return ""
    chars = []
    i = 0
    while i < len(x):
        if x[i] == "<":
            while i < len(x) and x[i] != ">":
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return "".join(chars)


def normalize_tokens(text: str) -> list[str]:
    out = []
    for token in text.split():
        token = _stripoff_tags(token.upper())
        if token:
            out.append(token)
    return out


def wer(refs: list[str], hyps: list[str]) -> float:
    ref_strs = [" ".join(normalize_tokens(r)) for r in refs]
    hyp_strs = [" ".join(normalize_tokens(h)) for h in hyps]
    return jiwer.wer(ref_strs, hyp_strs)
