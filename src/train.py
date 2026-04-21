import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

from data import load_datasets
from model import build_transformer
from seed import set_global_seed
from translate import sample_translations
from visualize import plot_encoder_self_attention


def run(args) -> keras.Model:
    set_global_seed(args.seed)

    train_ds, val_ds, test_pairs, spa_vec, eng_vec = load_datasets(
        batch_size=args.batch_size, seed=args.seed
    )

    transformer = build_transformer(mha_kind=args.mha)
    transformer.summary()
    transformer.compile(
        "rmsprop",
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=["accuracy"],
    )
    transformer.fit(train_ds, epochs=args.epochs, validation_data=val_ds)

    sample_translations(transformer, test_pairs, spa_vec, eng_vec)

    if args.visualize:
        if not args.sentence:
            raise ValueError("--visualize requires --sentence")
        out_path = plot_encoder_self_attention(
            sentence=args.sentence,
            transformer=transformer,
            spa_vec=spa_vec,
            head=args.head,
        )
        print(f"Saved attention heatmap to {out_path}")

    return transformer
