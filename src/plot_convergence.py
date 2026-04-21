"""Regenerate convergence plots from an existing run's metrics.csv.

Usage:
    python src/plot_convergence.py results/2026-04-21_15-53-04
    python src/plot_convergence.py results/run_a results/run_b -o comparison/
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot import plot_from_csv, read_metrics_csv


def _metrics_path(run_dir_or_csv: str) -> Path:
    p = Path(run_dir_or_csv)
    if p.is_file():
        return p
    return p / "metrics.csv"


def _overlay(run_paths: list[Path], out_dir: Path) -> None:
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    fig_acc, ax_acc = plt.subplots(figsize=(9, 5))
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5))

    for idx, csv_path in enumerate(run_paths):
        epochs, loss, acc, val_loss, val_acc = read_metrics_csv(csv_path)
        if not epochs:
            continue
        label = csv_path.parent.name
        color = colors[idx % len(colors)]
        ax_acc.plot(epochs, acc, color=color, linestyle="-", label=f"{label} train")
        ax_acc.plot(epochs, val_acc, color=color, linestyle="--", label=f"{label} val")
        ax_loss.plot(epochs, loss, color=color, linestyle="-", label=f"{label} train")
        ax_loss.plot(epochs, val_loss, color=color, linestyle="--", label=f"{label} val")

    for ax, ylabel, title in (
        (ax_acc, "accuracy", "Accuracy comparison"),
        (ax_loss, "loss", "Loss comparison"),
    ):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig_acc.tight_layout()
    fig_acc.savefig(out_dir / "comparison_accuracy.png", dpi=120)
    plt.close(fig_acc)
    fig_loss.tight_layout()
    fig_loss.savefig(out_dir / "comparison_loss.png", dpi=120)
    plt.close(fig_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Run directory or metrics.csv path(s).")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: run dir for single input, cwd for multi).")
    args = parser.parse_args()

    csv_paths = [_metrics_path(r) for r in args.runs]
    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        raise SystemExit(f"metrics.csv not found: {missing}")

    if len(csv_paths) == 1:
        csv_path = csv_paths[0]
        out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_from_csv(csv_path, out_dir)
        print(f"Plots saved to: {out_dir}")
        return

    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_paths:
        plot_from_csv(csv_path, csv_path.parent)
    _overlay(csv_paths, out_dir)
    print(f"Per-run plots updated; comparison saved to: {out_dir}")


if __name__ == "__main__":
    main()
