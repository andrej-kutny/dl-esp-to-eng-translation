# Assignment 2: Deep Learning VT2026

Transformer-based machine translation from Spanish to English, built on top of the Keras [Neural machine translation with a Transformer and Keras](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/) example.

The base code lives in [`src/Lab2_machine_translation-1.py`](src/Lab2_machine_translation-1.py) and was exported from a Jupyter notebook, so it still contains a number of `print` statements that can be cleaned up. Compared to the Keras example, the translation direction has been flipped (Spanish → English) and a few bugs that prevented the original example from running have been fixed.

All task-by-task changes are documented in [`changes.md`](changes.md).

## Background

The lab uses the same dataset and dependencies as the linked Keras example. You can download the dataset the same way the example does. The base code is configured for a single training epoch, which is handy while prototyping; once you are ready to record results you should increase the number of epochs. Working on a smaller subset of the data is allowed if training takes too long.

Suggested reading for Task 1 is the original [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper — in particular the section describing *Multi-Head Attention*. The [Keras attention MIL classification example](https://keras.io/examples/vision/attention_mil_classification/) is a good reference for how to structure a custom layer.

## Setup

```
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `keras`, `numpy`, `matplotlib`. The dataset is downloaded automatically on first run via `keras.utils.get_file`.

## Usage

All tasks run through a single CLI entrypoint, [`src/cli.py`](src/cli.py), which wraps the shared modules under `src/` (`data.py`, `attention.py`, `model.py`, `translate.py`, `train.py`, `visualize.py`, `seed.py`). The original notebook export ([`src/Lab2_machine_translation-1.py`](src/Lab2_machine_translation-1.py)) is kept untouched as reference; the earlier standalone per-task scripts have been removed in favor of the shared structure, see [`task1_task0_changes.md`](task1_task0_changes.md) for the bug fixes they contained.

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--mha {keras,custom}` | `keras` | Encoder attention implementation. `keras` uses `layers.MultiHeadAttention` (Task 0 baseline); `custom` uses the hand-rolled layer in [`src/attention.py`](src/attention.py) (Task 1). |
| `--epochs INT` | `30` | Training epochs. Paper-level convergence needs ~30. |
| `--batch-size INT` | `64` | Minibatch size. |
| `--seed INT` | `1337` | Seeds `random`, `numpy`, and TensorFlow for reproducibility. |
| `--visualize` | off | After training, plot encoder self-attention for `--sentence` (Task 2) — all 8 heads. |
| `--sentence STR` | — | Spanish input sentence to visualize. Required with `--visualize`. |

### Run artifacts

Each invocation creates `results/YYYY-MM-DD_HH-MM-SS/` at the repo root and writes:

- `config.json` — CLI args + model hyperparams (`embed_dim`, `latent_dim`, `num_heads`, `vocab_size`, `sequence_length`) + start timestamp.
- `summary.txt` — `model.summary()` output.
- `metrics.csv` — per-epoch `loss`, `accuracy`, `val_loss`, `val_accuracy` (via `keras.callbacks.CSVLogger`).
- `final_metrics.json` — last-epoch values + `epochs_trained`.
- `translations.txt` — the 30 sample translations printed at the end of training.
- `attention_head0.png` … `attention_head7.png` — one PNG per head (only with `--visualize`).
- `attention_meta.json` — input sentence, tokenization, and shape info (only with `--visualize`).

`results/` is gitignored; copy what you want into the report manually.

## Running each task

### Task 0 — Baseline

Keras `layers.MultiHeadAttention` in the encoder. This is the reference run for Task 1 to be compared against.

```
python src/cli.py --mha keras --epochs 30
```

### Task 1 — Custom `MultiHeadAttention`

Same architecture, but the encoder's attention is the hand-rolled implementation in [`src/attention.py`](src/attention.py).

```
python src/cli.py --mha custom --epochs 30
```

Run both with the same `--seed` to make the comparison fair.

### Task 2 — Attention visualization

Requires a trained model, so visualization piggybacks on a training run. The custom MHA layer returns attention scores when asked; [`src/visualize.py`](src/visualize.py) builds a secondary model from the trained encoder and saves one PNG per head (all 8) into the run directory.

```
python src/cli.py --mha custom --epochs 30 \
    --visualize --sentence "tom esta en la cocina"
```

Resulting PNGs should look similar to [`docs/img/image-2.png`](docs/img/image-2.png) — diagonal-ish attention on content tokens, near-zero on padding.

## Variations worth trying

All of the below reuse the same CLI, no code changes required:

- **Quick smoke test** before a full run — 1 epoch in each mode to confirm both code paths still work:
  ```
  python src/cli.py --mha keras  --epochs 1
  python src/cli.py --mha custom --epochs 1
  ```
- **Reproducibility / seed sensitivity**: repeat a run with different `--seed` values to check that the `keras` vs `custom` gap is within noise.
- **Batch-size effect on convergence / wall time**: `--batch-size 32` or `--batch-size 128` with the same epochs.
- **All attention heads** are saved per run (0–7) — no need to re-run for different heads.
- **Visualize a different sentence** by changing `--sentence` (Spanish only; the visualizer uses the Spanish vectorizer).
- **Cross-check the custom MHA** by visualizing attention in `--mha keras` mode too — `layers.MultiHeadAttention` also supports `return_attention_scores=True`, so the same plotting path works.

### Not exposed as CLI flags (edit source to change)

These are fixed as module constants to keep the CLI minimal; change them in code if you want to experiment:

- Model capacity (`EMBED_DIM`, `LATENT_DIM`, `NUM_HEADS`) — [`src/model.py`](src/model.py).
- Vocabulary / sequence length (`VOCAB_SIZE`, `SEQUENCE_LENGTH`) — [`src/data.py`](src/data.py).
- Dataset subset / validation fraction — currently 70/15/15 in [`src/data.py`](src/data.py) `_split_pairs()`.
- Checkpointing / model saving — not implemented; each run trains from scratch, which is also why `--visualize` piggybacks on training.

## Tasks

### Task 0 — Baseline

Get the example code to work as-is and record results with the base code. These numbers are the baseline that Task 1 will be compared against.

### Task 1 — Custom `MultiHeadAttention`

Implement your own `MultiHeadAttention` layer and replace `layers.MultiHeadAttention` in the transformer encoder. After swapping in your implementation, retrain the model on the same data and compare the results to Task 0.

### Task 2 — Attention visualization

Visualize the self-attention produced by your custom `MultiHeadAttention` layer and plot an attention matrix for a single sentence.

- Modify the layer to also return the attention scores and use those scores to build the matrix.
- Plotting a single head is enough.
- The resulting heatmap should look similar to the reference below.

![Self-attention heatmap, head 0](docs/img/image-2.png)

## References

- Keras example: <https://keras.io/examples/nlp/neural_machine_translation_with_transformer/>
- Custom layer reference: <https://keras.io/examples/vision/attention_mil_classification/>
- Vaswani et al., *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>
