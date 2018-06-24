#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention import AttentionModel as BaseAttentionModel
from .cells.CLSTMCell import CLSTMCell

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

    def load(self, sess, ckpt, flags):
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        return
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}

        loaded_kernel = tf.get_variable("loaded_kernel", shape=[1920, 2560])

        saver2 = tf.train.Saver(var_list={"decoder/attention_wrapper/basic_lstm_cell/kernel": loaded_kernel})
        saver2.restore(sess, ckpt)

        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope="decoder/attention_wrapper/basic_lstm_cell/kernel"):
            del var_list[var.op.name]

            if var.op.name == "decoder/attention_wrapper/basic_lstm_cell/kernel":
                print(sess.run(tf.pad(loaded_kernel, [[0, 35], [0, 0]])))
                print(var.get_shape())
                var = tf.assign(var, tf.pad(loaded_kernel, [[0, 35], [0, 0]]))

            print(sess.run(var))

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len), self.contexts, (self.target_labels, self.target_seq_len)) = \
                self._iterator.get_next()
        else:
            ((self.input_filenames, self.inputs, self.input_seq_len), self.contexts) = self._iterator.get_next()

    #def _get_decoder_cell(self):
        #return CLSTMCell(self.hparams.decoder_num_units, self.contexts)

    def _train_decode_fn(self, decoder_inputs, target_seq_len, initial_state, encoder_outputs, decoder_cell, scope):
        self.contexts = tf.tile(tf.expand_dims(self.contexts, axis=1), [1, tf.shape(decoder_inputs)[1], 1])
        decoder_inputs = tf.concat([decoder_inputs, self.contexts], axis=-1)
        return super()._train_decode_fn_default(decoder_inputs, target_seq_len,
                initial_state, encoder_outputs, decoder_cell, scope)

    def _eval_decode_fn(self, encoder_outputs, decoder_cell, scope):
        return super()._eval_decode_fn_default(encoder_outputs, decoder_cell, scope, context=self.contexts)
