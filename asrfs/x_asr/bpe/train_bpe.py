# asrfs/x_asr/bpe/train_bpe.py
"""离线训 LibriSpeech BPE-500(SentencePiece unigram),参数照抄 icefall
egs/librispeech/ASR/local/train_bpe_model.py。人工触发一次,产物 .model 入 git,
集群离线可用无下载。文本取 train split 转写、.lower()(与 prepare_ctc_example
编码一致)。

*** 偏差记录(controller 授权,2026-07-06)***
本产物只用 train.clean.100 一个 split(~2.8 万行)训练,而非原设计的三 split
全量(~28 万行)。原因:复用 common.full_data._stream_split 只为读 text,却会
拉取 HF 上整份 audio parquet row-group,~280k 行需下载数十 GB;实测流式速率
~28 行/秒,三 split 全量在本机需 ~2.8 小时才能开始 spm 训练,不可接受。
500 piece 的 unigram 词表对 100h-vs-960h 文本不敏感(同为 LibriSpeech 朗读英语,
词表覆盖近乎一致),对 icefall 的可比性实际不受影响。icefall spm 参数
(unigram / vocab 500 / coverage 1.0 / user_defined_symbols / unk 2 / bos eos -1)
与 .lower() 均保持不变。若日后需三 split 全量重训,把 TRAIN_SPLITS_FOR_BPE
改回 TRAIN_SPLITS 即可。

用法:
    python -m asrfs.x_asr.bpe.train_bpe            # train.clean.100(~2.8 万行)
    python -m asrfs.x_asr.bpe.train_bpe --subset-head 500   # 冒烟(词表不足 500 会报错)
"""

import argparse
import tempfile
from pathlib import Path

import sentencepiece as spm

from asrfs.common.full_data import TRAIN_SPLITS, _stream_split

BPE_DIR = Path(__file__).resolve().parent
MODEL_PREFIX = BPE_DIR / "librispeech_bpe500"
VOCAB_SIZE = 500

# 偏差(见模块 docstring):只用 clean.100。恢复全量把这行改成 TRAIN_SPLITS。
TRAIN_SPLITS_FOR_BPE = tuple(s for s in TRAIN_SPLITS if s[2] == "train.clean.100")


def _iter_train_text(subset_head=None):
    for config, split, _name in TRAIN_SPLITS_FOR_BPE:
        for row in _stream_split(config, split, subset_head=subset_head):
            yield row["text"].lower()


def train(subset_head=None) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        text_path = f.name
        for line in _iter_train_text(subset_head):
            f.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=str(MODEL_PREFIX),
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        input_sentence_size=100000000,
        user_defined_symbols=["<blk>", "<sos/eos>"],
        unk_id=2,
        bos_id=-1,
        eos_id=-1,
    )
    Path(text_path).unlink(missing_ok=True)
    return MODEL_PREFIX.with_suffix(".model")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset-head", type=int, default=None)
    args = p.parse_args()
    out = train(args.subset_head)
    sp = spm.SentencePieceProcessor(model_file=str(out))
    print(f"[OK] {out} piece_size={sp.get_piece_size()}")
