import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import pathlib
import random
import re
import string

import keras
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras.layers import TextVectorization


VOCAB_SIZE = 15000
SEQUENCE_LENGTH = 20

_STRIP_CHARS = string.punctuation + "¿"
_STRIP_CHARS = _STRIP_CHARS.replace("[", "").replace("]", "")


def _resolve_spa_txt() -> pathlib.Path:
    archive = keras.utils.get_file(
        fname="spa-eng.zip",
        origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        extract=True,
    )
    parent = pathlib.Path(archive).parent
    candidates = [
        parent / "spa-eng_extracted" / "spa-eng" / "spa.txt",
        parent / "spa-eng" / "spa.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find spa.txt. Tried: " + ", ".join(str(c) for c in candidates)
    )


def _load_pairs(path: pathlib.Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    pairs: list[tuple[str, str]] = []
    for line in lines:
        eng, spa = line.split("\t")
        eng = "[start] " + eng + " [end]"
        pairs.append((spa, eng))
    return pairs


def _split_pairs(
    pairs: list[tuple[str, str]],
) -> tuple[list, list, list]:
    random.shuffle(pairs)
    num_val = int(0.15 * len(pairs))
    num_train = len(pairs) - 2 * num_val
    train = pairs[:num_train]
    val = pairs[num_train : num_train + num_val]
    test = pairs[num_train + num_val :]
    return train, val, test


def _custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(
        lowercase, "[%s]" % re.escape(_STRIP_CHARS), ""
    )


def _build_vectorizers(
    train_pairs: list[tuple[str, str]],
) -> tuple[TextVectorization, TextVectorization]:
    eng_vec = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH + 1,
        standardize=_custom_standardization,
    )
    spa_vec = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize=_custom_standardization,
    )
    eng_vec.adapt([pair[1] for pair in train_pairs])
    spa_vec.adapt([pair[0] for pair in train_pairs])
    return spa_vec, eng_vec


def _format_dataset(eng, spa, eng_vec, spa_vec):
    eng = eng_vec(eng)
    spa = spa_vec(spa)
    return (
        {"encoder_inputs": spa, "decoder_inputs": eng[:, :-1]},
        eng[:, 1:],
    )


def _make_dataset(pairs, batch_size, eng_vec, spa_vec):
    spa_texts, eng_texts = zip(*pairs)
    ds = tf_data.Dataset.from_tensor_slices((list(eng_texts), list(spa_texts)))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda e, s: _format_dataset(e, s, eng_vec, spa_vec))
    return ds.cache().shuffle(2048).prefetch(16)


def load_datasets(batch_size: int, seed: int):
    random.seed(seed)
    path = _resolve_spa_txt()
    pairs = _load_pairs(path)
    train_pairs, val_pairs, test_pairs = _split_pairs(pairs)

    print(f"{len(pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

    spa_vec, eng_vec = _build_vectorizers(train_pairs)
    train_ds = _make_dataset(train_pairs, batch_size, eng_vec, spa_vec)
    val_ds = _make_dataset(val_pairs, batch_size, eng_vec, spa_vec)
    return train_ds, val_ds, test_pairs, spa_vec, eng_vec
