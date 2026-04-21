import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import random

import numpy as np
import tensorflow as tf


MAX_DECODED_SENTENCE_LENGTH = 20


def make_decoder(transformer, spa_vec, eng_vec):
    eng_vocab = eng_vec.get_vocabulary()
    eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))

    def decode_sequence(input_sentence: str) -> str:
        tokenized_input = spa_vec([input_sentence])
        decoded_sentence = "[start]"
        for i in range(MAX_DECODED_SENTENCE_LENGTH):
            tokenized_target = eng_vec([decoded_sentence])[:, :-1]
            predictions = transformer(
                {
                    "encoder_inputs": tokenized_input,
                    "decoder_inputs": tokenized_target,
                }
            )
            sampled_token_index = int(np.argmax(predictions[0, i, :]))
            sampled_token = eng_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if sampled_token == "[end]":
                break
        return decoded_sentence

    return decode_sequence


def sample_translations(transformer, test_pairs, spa_vec, eng_vec, n: int = 30):
    decode_sequence = make_decoder(transformer, spa_vec, eng_vec)
    test_spa_texts = [pair[0] for pair in test_pairs]
    for _ in range(n):
        input_sentence = random.choice(test_spa_texts)
        translated = decode_sequence(input_sentence)
        print(f"{input_sentence} = {translated}")
