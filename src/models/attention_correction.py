#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base import BaseModel

import tensorflow as tf
import numpy as np

from tensorflow.python.layers import core as layers_core

from six.moves import xrange as range

from csp.utils import ops_utils, model_utils
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from .attention import AttentionModel as BaseAttentionModel

# Hyper-parameters
SAMPLING_TEMPERATURE = 0

max_gradient_norm = 5.0

class AttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__()

    def __call__(self, hparams, mode, iterator, **kwargs):
        return BaseAttentionModel.__call__(self, hparams, mode, iterator, trainable_scope="correction")

    def _build_decoder(self, encoder_outputs, encoder_final_state):
        logits, sample_ids, final_context_state = super()._build_decoder(encoder_outputs, encoder_final_state)

        ENCODER_NUM_UNITS = 640
        DECODER_NUM_UNITS = 320
        ATTENTION_LAYER_SIZE = 256
        with tf.variable_scope('correction') as correction_scope:
            correction_encoder_outputs, correction_encoder_state = tf.nn.dynamic_rnn(
                model_utils.single_cell("lstm", ENCODER_NUM_UNITS, self.mode),
                self.decoder_emb_layer(logits), sequence_length=self.target_seq_len,
                initial_state=final_context_state.cell_state
            )

            correction_decoder = model_utils.single_cell("lstm", DECODER_NUM_UNITS, self.mode)
            attention_mechanism2 = tf.contrib.seq2seq.LuongAttention(
                DECODER_NUM_UNITS,
                memory=correction_encoder_outputs,
                scale=True,
                memory_sequence_length=self.target_seq_len
            )

            correction_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                correction_decoder, attention_mechanism2,
                attention_layer_size=ATTENTION_LAYER_SIZE,
                alignment_history=False,
                output_attention=True
            )

            if self.train_mode:
                decoder_emb_inp = self.decoder_emb_layer(self.targets)
                helper2 = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.target_seq_len)
                decoder2 = tf.contrib.seq2seq.BasicDecoder(
                    correction_decoder_cell,
                    helper2,
                    correction_decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
                )

                outputs2, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder2, swap_memory=True, scope=correction_scope)

                logits = self.output_layer(outputs2.rnn_output)
                sample_ids = tf.argmax(logits, axis=-1)
            else:
                helper2 = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    lambda ids: self.decoder_emb_layer(tf.one_hot(ids, depth=self.hparams.num_classes)),
                    start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                    end_token=self.hparams.eos_index
                )

                decoder2 = tf.contrib.seq2seq.BasicDecoder(
                    correction_decoder_cell,
                    helper2,
                    correction_decoder_cell.zero_state(self.batch_size, dtype=tf.float32),
                    output_layer=self.output_layer
                )

                outputs2, final_context_state2, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder2,
                    swap_memory=True,
                    maximum_iterations=100,
                    scope=correction_scope
                )

                sample_ids = outputs2.sample_id
                logits = outputs2.rnn_output

        return logits, sample_ids, final_context_state