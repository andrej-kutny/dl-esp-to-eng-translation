import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np


def _compute_encoder_scores(transformer: keras.Model, tokenized):
    encoder_pos_emb = transformer.get_layer("encoder_pos_emb")
    encoder_block = transformer.get_layer("encoder_block")
    embedded = encoder_pos_emb(tokenized)
    mask = encoder_pos_emb.compute_mask(tokenized)
    _, scores = encoder_block(embedded, mask=mask, return_attention_scores=True)
    return scores


def _tokens_for(sentence: str, spa_vec) -> tuple[list[str], int]:
    token_ids = np.asarray(spa_vec([sentence]))[0]
    vocab = spa_vec.get_vocabulary()
    tokens = [vocab[i] if i < len(vocab) else "" for i in token_ids]
    nonzero = [idx for idx, i in enumerate(token_ids) if i != 0]
    content_len = (max(nonzero) + 1) if nonzero else len(tokens)
    return tokens, content_len


def _plot_head(head_scores, tokens, content_len, head: int, out_path: Path) -> None:
    trimmed = head_scores[:content_len, :content_len]
    trimmed_tokens = tokens[:content_len]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trimmed, cmap="viridis", aspect="auto")
    ax.set_xticks(range(content_len))
    ax.set_yticks(range(content_len))
    ax.set_xticklabels(trimmed_tokens, rotation=45, ha="right")
    ax.set_yticklabels(trimmed_tokens)
    ax.set_xlabel("key tokens")
    ax.set_ylabel("query tokens")
    ax.set_title(f"Encoder self-attention (head {head})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_encoder_heads(
    sentence: str,
    transformer: keras.Model,
    spa_vec,
    out_dir: Path,
    filename_fmt: str = "attention_head{head}.png",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenized = spa_vec([sentence])
    scores = np.asarray(_compute_encoder_scores(transformer, tokenized))
    num_heads = scores.shape[1]

    tokens, content_len = _tokens_for(sentence, spa_vec)

    meta = {
        "sentence": sentence,
        "tokens": tokens,
        "content_length": int(content_len),
        "num_heads": int(num_heads),
        "scores_shape": list(scores.shape),
    }
    (out_dir / "attention_meta.json").write_text(json.dumps(meta, indent=2))

    paths: list[Path] = []
    for head in range(num_heads):
        path = out_dir / filename_fmt.format(head=head)
        _plot_head(scores[0, head], tokens, content_len, head, path)
        paths.append(path)
    return paths
