import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf
from keras import layers

from attention import MultiHeadAttention as CustomMultiHeadAttention
from data import SEQUENCE_LENGTH, VOCAB_SIZE


EMBED_DIM = 256
LATENT_DIM = 2048
NUM_HEADS = 8


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, mha_kind="keras", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.mha_kind = mha_kind
        if mha_kind == "keras":
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
        elif mha_kind == "custom":
            self.attention = CustomMultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
        else:
            raise ValueError(f"Unknown mha_kind: {mha_kind!r}")
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None, return_attention_scores=False):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        if return_attention_scores:
            attention_output, attention_scores = self.attention(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=padding_mask,
                return_attention_scores=True,
            )
        else:
            attention_output = self.attention(
                query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
            )
            attention_scores = None

        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        encoded = self.layernorm_2(proj_input + proj_output)
        if return_attention_scores:
            return encoded, attention_scores
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
                "mha_kind": self.mha_kind,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.supports_masking = True

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, name="self_attention"
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, name="cross_attention"
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ],
            name="dense_proj",
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]

        causal_mask = self.get_causal_attention_mask(decoder_inputs)

        decoder_padding_mask = None
        encoder_padding_mask = None
        if mask is not None:
            if isinstance(mask, (list, tuple)):
                if len(mask) > 0:
                    decoder_padding_mask = mask[0]
                if len(mask) > 1:
                    encoder_padding_mask = mask[1]
            else:
                decoder_padding_mask = mask

        self_attn_mask = tf.cast(causal_mask, "int32")
        if decoder_padding_mask is not None:
            pad = tf.cast(decoder_padding_mask[:, None, :], dtype="int32")
            self_attn_mask = tf.minimum(self_attn_mask, pad)

        attn_output_1 = self.attention_1(
            query=decoder_inputs,
            value=decoder_inputs,
            key=decoder_inputs,
            attention_mask=self_attn_mask,
        )
        out_1 = self.layernorm_1(decoder_inputs + attn_output_1)

        cross_attn_mask = None
        if encoder_padding_mask is not None:
            cross_attn_mask = tf.cast(encoder_padding_mask[:, None, :], dtype="int32")

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attn_mask,
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(i >= j, dtype=tf.bool)

        mask = tf.expand_dims(mask, axis=0)
        mask = tf.broadcast_to(mask, [batch_size, seq_len, seq_len])
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


def build_transformer(mha_kind: str = "keras") -> keras.Model:
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_pos_emb = PositionalEmbedding(
        SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM, name="encoder_pos_emb"
    )
    encoder_block = TransformerEncoder(
        EMBED_DIM, LATENT_DIM, NUM_HEADS, mha_kind=mha_kind, name="encoder_block"
    )
    x = encoder_pos_emb(encoder_inputs)
    encoder_outputs = encoder_block(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(
        shape=(None, EMBED_DIM), name="decoder_state_inputs"
    )

    x = PositionalEmbedding(
        SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM, name="decoder_pos_emb"
    )(decoder_inputs)
    x = TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS, name="decoder_block")(
        [x, encoded_seq_inputs]
    )
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)

    decoder = keras.Model(
        [decoder_inputs, encoded_seq_inputs], decoder_outputs, name="decoder"
    )

    final_decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    transformer = keras.Model(
        {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
        final_decoder_outputs,
        name="transformer",
    )
    return transformer
