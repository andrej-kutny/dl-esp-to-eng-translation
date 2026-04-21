import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lab2 Spanish→English transformer training + attention visualization.",
    )
    parser.add_argument(
        "--mha",
        choices=["keras", "custom"],
        default="keras",
        help="Encoder attention implementation: 'keras' (layers.MultiHeadAttention) or 'custom' (hand-rolled).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (30 reproduces task0 baseline).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for python/numpy/tensorflow.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After training, plot encoder self-attention heatmap for --sentence and exit normally.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=None,
        help="Spanish input sentence to visualize; required with --visualize.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.visualize and not args.sentence:
        raise SystemExit("--visualize requires --sentence")

    from train import run

    run(args)


if __name__ == "__main__":
    main()
