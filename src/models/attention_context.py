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


class AttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(
            train_decode_fn=self._train_decode_fn,
            eval_decode_fn=self._eval_decode_fn
        )

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}

        del var_list["Variable"]
        del var_list["Variable_1"]
        del var_list["context_embedding/bias"]
        del var_list["context_embedding/kernel"]

        loaded_kernel = tf.get_variable("loaded_kernel", shape=[1536, 2048])

        saver2 = tf.train.Saver(var_list={"decoder/attention_wrapper/basic_lstm_cell/kernel": loaded_kernel})
        saver2.restore(sess, ckpt)

        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope="decoder/attention_wrapper/basic_lstm_cell/kernel"):
            del var_list[var.op.name]

            if var.op.name == "decoder/attention_wrapper/basic_lstm_cell/kernel":
                var = tf.assign(
                    var,
                    tf.concat([loaded_kernel[:512], tf.zeros([128, 2048]),
                               loaded_kernel[512:]], axis=0))

            sess.run(var)

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    def _assign_input(self):
        self.dlg_ids, (self.inputs, self.input_seq_len), (
            self.targets, self.target_seq_len), self.da_labels = self.iterator.get_next()
        self.contexts_one_hot = tf.one_hot(self.da_labels, 43)
        self.contexts = tf.layers.dense(self.contexts_one_hot, 128, name="context_embedding")

    def _train_decode_fn(self, decoder_inputs, target_seq_len, initial_state, encoder_outputs, decoder_cell, scope):
        return super()._train_decode_fn_default(
            decoder_inputs, target_seq_len,
            initial_state, encoder_outputs, decoder_cell, scope,
            context=self.contexts)

    def _eval_decode_fn(self, initial_state, encoder_outputs, decoder_cell, scope):
        return super()._eval_decode_fn_default(
            initial_state, encoder_outputs, decoder_cell, scope,
            context=self.contexts)

    @classmethod
    def trainable_variables(cls):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
