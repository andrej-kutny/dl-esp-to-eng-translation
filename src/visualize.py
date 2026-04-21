import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from pathlib import Path

import keras
import matplotlib.pyplot as plt
from keras import ops


DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"


def _build_scores_model(transformer: keras.Model) -> keras.Model:
    encoder_inputs = transformer.get_layer("encoder_pos_emb").input
    if encoder_inputs is None:
        encoder_inputs = transformer.input["encoder_inputs"]
    encoder_pos_emb = transformer.get_layer("encoder_pos_emb")
    encoder_block = transformer.get_layer("encoder_block")
    embedded = encoder_pos_emb(encoder_inputs)
    _, scores = encoder_block(embedded, return_attention_scores=True)
    return keras.Model(inputs=encoder_inputs, outputs=scores)


def _tokens_for(sentence: str, spa_vec) -> tuple[list[str], int]:
    token_ids = ops.convert_to_numpy(spa_vec([sentence]))[0]
    vocab = spa_vec.get_vocabulary()
    tokens = [vocab[i] if i < len(vocab) else "" for i in token_ids]
    nonzero = [idx for idx, i in enumerate(token_ids) if i != 0]
    content_len = (max(nonzero) + 1) if nonzero else len(tokens)
    return tokens, content_len


def plot_encoder_self_attention(
    sentence: str,
    transformer: keras.Model,
    spa_vec,
    head: int = 0,
    out_path: Path | None = None,
) -> Path:
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / f"attention_head{head}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scores_model = _build_scores_model(transformer)
    tokenized = spa_vec([sentence])
    scores = ops.convert_to_numpy(scores_model(tokenized))
    head_scores = scores[0, head]

    tokens, content_len = _tokens_for(sentence, spa_vec)
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
    return out_path
