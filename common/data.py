"""LibriSpeech loading helpers shared by all four models.

Smoke stage: stream the head of train.clean.100 once and cache it to disk,
so reruns never touch the network. Full 960h precompute comes later.
"""

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

LIBRISPEECH_REPO = "openslr/librispeech_asr"


def fetch_smoke_subset(n: int = 128) -> Dataset:
    """First n utterances of train.clean.100: columns audio_array / sampling_rate / text / id."""
    cache = DATA_DIR / f"smoke_train.clean.100_head{n}"
    if cache.exists():
        return load_from_disk(str(cache))

    stream = load_dataset(LIBRISPEECH_REPO, "clean", split="train.100", streaming=True)
    # decode=False sidesteps datasets 5.x's torchcodec/ffmpeg audio path;
    # soundfile reads FLAC natively
    stream = stream.cast_column("audio", Audio(decode=False))
    rows = []
    for sample in stream.take(n):
        array, sampling_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        rows.append(
            {
                "id": sample["id"],
                "audio_array": np.asarray(array),
                "sampling_rate": sampling_rate,
                "text": sample["text"],
            }
        )
    ds = Dataset.from_list(rows)
    DATA_DIR.mkdir(exist_ok=True)
    ds.save_to_disk(str(cache))
    return ds


if __name__ == "__main__":
    ds = fetch_smoke_subset(n=8)
    first = ds[0]
    dur = len(first["audio_array"]) / first["sampling_rate"]
    print(f"cached {len(ds)} samples; first: id={first['id']} dur={dur:.1f}s sr={first['sampling_rate']}")
    print(f"text: {first['text']}")
