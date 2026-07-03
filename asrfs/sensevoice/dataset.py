"""sensevoice 数据面:CTC 族共享实现在 asrfs/common/ctc.py,此处只留契约名。"""

from asrfs.common.ctc import (
    CTCCollator,
    build_ctc_dataset,
    ctc_decode,
    ctc_greedy_decode,
    prepare_ctc_example,
)


def prepare_example(sample: dict, feature_extractor, tokenizer) -> dict:
    return prepare_ctc_example(sample, feature_extractor, tokenizer)


def make_example(processor, audio, sampling_rate: int, text: str) -> dict:
    sample = {"audio_array": audio, "sampling_rate": sampling_rate, "text": text}
    return prepare_example(sample, processor.feature_extractor, processor.tokenizer)


def build_collator(cfg: dict, processor, model) -> CTCCollator:
    # blank(= CTC pad)从 processor 自取;model 参数契约占位(CE 族才用)。
    return CTCCollator(
        processor.feature_extractor, pad_label_id=processor.tokenizer.vocab_size
    )


def build_dataset(cfg: dict, processor, mode: str) -> tuple:
    return build_ctc_dataset(cfg, processor, mode, model_name="sensevoice")


def decode(model, processor, batch) -> list[str]:
    return ctc_decode(model, processor, batch)
