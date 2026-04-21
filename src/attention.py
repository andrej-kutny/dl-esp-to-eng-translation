import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras.ops as ops
from keras import layers


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        if key_dim % num_heads != 0:
            raise ValueError("key_dim must be divisible by num_heads.")
        self.projection_dim = key_dim // num_heads

        self.query_dense = layers.Dense(key_dim)
        self.key_dense = layers.Dense(key_dim)
        self.value_dense = layers.Dense(key_dim)
        self.output_dense = layers.Dense(key_dim)

    def _split_heads(self, inputs):
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        inputs = ops.reshape(
            inputs, (batch_size, seq_len, self.num_heads, self.projection_dim)
        )
        return ops.transpose(inputs, (0, 2, 1, 3))

    def _combine_heads(self, inputs):
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[2]
        inputs = ops.transpose(inputs, (0, 2, 1, 3))
        return ops.reshape(inputs, (batch_size, seq_len, self.key_dim))

    def call(
        self,
        query,
        value,
        key,
        attention_mask=None,
        return_attention_scores=False,
    ):
        query = self._split_heads(self.query_dense(query))
        key = self._split_heads(self.key_dense(key))
        value = self._split_heads(self.value_dense(value))

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.projection_dim, scores.dtype))

        if attention_mask is not None:
            mask = ops.cast(attention_mask, "bool")
            if len(mask.shape) == 2:
                mask = ops.expand_dims(mask, axis=1)
                mask = ops.expand_dims(mask, axis=1)
            elif len(mask.shape) == 3:
                mask = ops.expand_dims(mask, axis=1)
            large_negative = ops.cast(-1e9, scores.dtype)
            scores = ops.where(mask, scores, large_negative)

        attention_weights = ops.softmax(scores, axis=-1)
        attention_output = ops.matmul(attention_weights, value)
        attention_output = self._combine_heads(attention_output)
        output = self.output_dense(attention_output)
        if return_attention_scores:
            return output, attention_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config
