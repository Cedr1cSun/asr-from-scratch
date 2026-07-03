"""CTC 族共享件(asrfs/common/ctc.py)与 parakeet 薄 re-export 的同一性。"""


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
