#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention import AttentionModel as BaseAttentionModel

"""Example soft monotonic alignment decoder implementation.
This file contains an example TensorFlow implementation of the approach
described in ``Online and Linear-Time Attention by Enforcing Monotonic
Alignments''.  The function monotonic_attention covers the algorithms in the
paper and should be general-purpose.  monotonic_alignment_decoder can be used
directly in place of tf.nn.seq2seq.attention_decoder.  This implementation
attempts to deviate as little as possible from tf.nn.seq2seq.attention_decoder,
in order to facilitate comparison between the two decoders.
"""
import tensorflow as tf


class LuongMonotonicAttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(attention_fn=self._attention_fn)

    def _attention_fn(self, encoder_outputs):
        return tf.contrib.seq2seq.LuongMonotonicAttention(
            self.hparams.decoder_num_units,
            encoder_outputs,
            memory_sequence_length=self.input_seq_len,
            scale=self.hparams.attention_energy_scale,
            score_bias_init=-4.0,
            sigmoid_noise=1,
            mode='hard'
        )

class BahdanauMonotonicAttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(attention_fn=self._attention_fn)

    def _attention_fn(self, encoder_outputs):
        return tf.contrib.seq2seq.BahdanauMonotonicAttention(
            self.hparams.decoder_num_units,
            encoder_outputs,
            memory_sequence_length=self.input_seq_len,
            score_bias_init=-3.0,
            sigmoid_noise=1,
            mode='hard'
        )