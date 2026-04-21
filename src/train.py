import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import contextlib

import keras

from data import SEQUENCE_LENGTH, VOCAB_SIZE, load_datasets
from model import EMBED_DIM, LATENT_DIM, NUM_HEADS, build_transformer
from plot import ConvergencePlotCallback
from run_dir import (
    create_run_dir,
    write_config,
    write_final_metrics,
    write_model_summary,
)
from seed import set_global_seed
from translate import sample_translations
from visualize import plot_all_encoder_heads


def run(args) -> keras.Model:
    set_global_seed(args.seed)

    run_dir = create_run_dir()
    print(f"Run directory: {run_dir}")

    write_config(
        run_dir,
        args,
        model_constants={
            "embed_dim": EMBED_DIM,
            "latent_dim": LATENT_DIM,
            "num_heads": NUM_HEADS,
            "vocab_size": VOCAB_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
        },
    )

    train_ds, val_ds, test_pairs, spa_vec, eng_vec = load_datasets(
        batch_size=args.batch_size, seed=args.seed
    )

    transformer = build_transformer(mha_kind=args.mha)
    transformer.summary()
    write_model_summary(run_dir, transformer)

    transformer.compile(
        "rmsprop",
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=["accuracy"],
    )
    history = transformer.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[
            keras.callbacks.CSVLogger(str(run_dir / "metrics.csv")),
            ConvergencePlotCallback(run_dir),
        ],
    )
    write_final_metrics(run_dir, history)

    translations_path = run_dir / "translations.txt"
    with open(translations_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(
        _Tee(f)
    ):
        sample_translations(transformer, test_pairs, spa_vec, eng_vec)

    if args.visualize:
        if not args.sentence:
            raise ValueError("--visualize requires --sentence")
        paths = plot_all_encoder_heads(
            sentence=args.sentence,
            transformer=transformer,
            spa_vec=spa_vec,
            out_dir=run_dir,
        )
        print(f"Saved {len(paths)} attention heatmaps to {run_dir}")

    print(f"Run artifacts saved to {run_dir}")
    return transformer


class _Tee:
    def __init__(self, file):
        self._file = file
        self._stdout = __import__("sys").stdout

    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)

    def flush(self):
        self._file.flush()
        self._stdout.flush()
