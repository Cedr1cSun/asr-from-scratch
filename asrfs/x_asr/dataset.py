"""x_asr 数据面:batch 面与 CTC 族完全一致(features/mask/labels-pad-1024),
collator/smoke 数据集直接复用 asrfs/common/ctc.py;解码走 RNN-T greedy。"""

from asrfs.common.ctc import CTCCollator, build_ctc_dataset, prepare_ctc_example


def prepare_example(sample: dict, feature_extractor, tokenizer) -> dict:
    return prepare_ctc_example(sample, feature_extractor, tokenizer)


def make_example(processor, audio, sampling_rate: int, text: str) -> dict:
    sample = {"audio_array": audio, "sampling_rate": sampling_rate, "text": text}
    return prepare_example(sample, processor.feature_extractor, processor.tokenizer)


def build_collator(cfg: dict, processor, model) -> CTCCollator:
    # RNN-T 的 blank/pad 语义与 CTC 族一致(= tokenizer.vocab_size);model 占位。
    return CTCCollator(
        processor.feature_extractor, pad_label_id=processor.tokenizer.vocab_size
    )


def build_dataset(cfg: dict, processor, mode: str) -> tuple:
    return build_ctc_dataset(cfg, processor, mode, model_name="x_asr")


def decode(model, processor, batch) -> list[str]:
    """契约解码:RNN-T greedy(模型内实现)→ detokenize。忽略 labels 等训练键。"""
    device = next(model.parameters()).device
    feats = batch["input_features"].to(device)
    mask = batch.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    hyps = model.greedy_decode(feats, mask)
    return [
        processor.tokenizer.decode(h, skip_special_tokens=True) if h else ""
        for h in hyps
    ]
