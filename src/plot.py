import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras


def read_metrics_csv(path: Path):
    epochs, loss, acc, val_loss, val_acc = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]) + 1)
            loss.append(float(row["loss"]))
            acc.append(float(row["accuracy"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))
    return epochs, loss, acc, val_loss, val_acc


def _best_checkpoints(epochs, val_acc):
    """Staircase of (epoch, val_acc) where val_acc reached a new high."""
    best = []
    best_so_far = float("-inf")
    for ep, v in zip(epochs, val_acc):
        if v > best_so_far:
            best_so_far = v
            best.append((ep, v))
    return best


def _plot_accuracy(epochs, acc, val_acc, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, acc, label="train", color="tab:blue")
    ax.plot(epochs, val_acc, label="val", color="tab:orange")

    checkpoints = _best_checkpoints(epochs, val_acc)
    if checkpoints:
        xs = [e for e, _ in checkpoints]
        ys = [v for _, v in checkpoints]
        ax.plot(xs, ys, color="tab:orange", linewidth=0.8, linestyle="--",
                label="val best")
        for ep, v in checkpoints:
            ax.plot(ep, v, "o", color="tab:orange", markersize=5)
            ax.annotate(f"{ep}: {v:.4f}", xy=(ep, v), xytext=(0, 6),
                        textcoords="offset points", rotation=45,
                        fontsize=7, color="tab:orange")

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_loss(epochs, loss, val_loss, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss, label="train", color="tab:blue")
    ax.plot(epochs, val_loss, label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_from_csv(metrics_csv: Path, out_dir: Path) -> None:
    metrics_csv = Path(metrics_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs, loss, acc, val_loss, val_acc = read_metrics_csv(metrics_csv)
    if not epochs:
        return
    _plot_accuracy(epochs, acc, val_acc, out_dir / "convergence_accuracy.png")
    _plot_loss(epochs, loss, val_loss, out_dir / "convergence_loss.png")


class ConvergencePlotCallback(keras.callbacks.Callback):
    """After every epoch: refresh convergence_{accuracy,loss}.png and append
    timing info to epoch_times.json in the run directory. When a sentence +
    vectorizer are provided, also snapshot encoder attention heatmaps on
    every new best val_accuracy into attention/{epoch}_{val_acc:.6f}/."""

    def __init__(self, run_dir: Path, sentence: str | None = None, spa_vec=None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.times_path = self.run_dir / "epoch_times.json"
        self.sentence = sentence
        self.spa_vec = spa_vec
        self._epochs: list[int] = []
        self._loss: list[float] = []
        self._acc: list[float] = []
        self._val_loss: list[float] = []
        self._val_acc: list[float] = []
        self._epoch_start_iso: str | None = None
        self._epoch_start_mono: float | None = None
        self._best_val_acc: float = float("-inf")
        self.times_path.write_text("{}")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_iso = datetime.now().isoformat()
        self._epoch_start_mono = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.monotonic() - (self._epoch_start_mono or time.monotonic())

        ep = epoch + 1
        val_acc = float(logs.get("val_accuracy", float("nan")))
        self._epochs.append(ep)
        self._loss.append(float(logs.get("loss", float("nan"))))
        self._acc.append(float(logs.get("accuracy", float("nan"))))
        self._val_loss.append(float(logs.get("val_loss", float("nan"))))
        self._val_acc.append(val_acc)

        times = json.loads(self.times_path.read_text() or "{}")
        times[str(ep)] = {
            "start": self._epoch_start_iso,
            "end": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 3),
        }
        self.times_path.write_text(json.dumps(times, indent=2))

        _plot_accuracy(
            self._epochs, self._acc, self._val_acc,
            self.run_dir / "convergence_accuracy.png",
        )
        _plot_loss(
            self._epochs, self._loss, self._val_loss,
            self.run_dir / "convergence_loss.png",
        )

        if (
            self.sentence
            and self.spa_vec is not None
            and val_acc == val_acc  # skip NaN
            and val_acc > self._best_val_acc
        ):
            self._best_val_acc = val_acc
            from visualize import plot_all_encoder_heads

            out_dir = self.run_dir / "attention" / f"{ep}_{val_acc:.6f}"
            plot_all_encoder_heads(
                sentence=self.sentence,
                transformer=self.model,
                spa_vec=self.spa_vec,
                out_dir=out_dir,
                filename_fmt="head_{head}.png",
            )
