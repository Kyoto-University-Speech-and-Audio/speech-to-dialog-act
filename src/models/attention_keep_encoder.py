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
        super().__init__()

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}
        del var_list["Variable"]
        del var_list["Variable_1"]
        #new_var_list = {}
        #for name in var_list:
        #    if name[:7] == "decoder":
        #        new_var_list[name] = var_list[name]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    @classmethod
    def trainable_variables(cls):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")