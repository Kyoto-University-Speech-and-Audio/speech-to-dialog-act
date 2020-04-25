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
#from .attentions.attention_wrapper import AttentionWrapper
from tensorflow.contrib.seq2seq import AttentionWrapper

NUM_SPEAKERS = 2

class AttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(
            attention_wrapper_fn=self._attention_wrapper
        )

    '''
    def _attention_wrapper(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 initial_cell_state=None, output_attention=False,
                 name=None):
        return AttentionWrapper(
            cell, attention_mechanism,
            #context=self.full_context,
            attention_layer_size=self.hparams.attention_layer_size,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            initial_cell_state=initial_cell_state,
            name=name
        )
    '''

    @classmethod
    def load(self, sess, ckpt, flags):
        #saver = tf.train.Saver()
        #saver.restore(sess, ckpt)
        #return
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}

        del var_list["Variable_1"]
        for s in range(NUM_SPEAKERS): del var_list["context_speaker_%d" % s]
        for name in var_list:
            if name[:36] == "decoder/contextual_attention_wrapper":
                var = var_list[name]
                del var_list[name]
                var_list["decoder/attention_wrapper" + name[36:]] = var

        print(var_list.keys())

        loaded_kernel = tf.get_variable("loaded_kernel", shape=[1920, 2560])

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

        return
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

    def get_extra_ops(self):
        return self._extra_ops

    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len),
             self.is_first_utts, self.speakers,
             (self.target_labels, self.target_seq_len)) = \
                self._iterator.get_next()
        else:
            ((self.input_filenames, self.inputs, self.input_seq_len),
             self.is_first_utts, self.speakers) = self._iterator.get_next()

    def _build_graph(self):
        self.contexts = []
        for s in range(NUM_SPEAKERS):
            self.contexts.append(
                tf.Variable(
                    self._get_decoder_cell().zero_state(self.hparams.batch_size, dtype=tf.float32),
                    name="context_speaker_%d" % s,
                    trainable=False, dtype=tf.float32))

        print(self.contexts)
        self.full_context = tf.concat(self.contexts, axis=-1)

        ret = super()._build_graph()

        print(self.final_context_state.cell_state)
        with tf.control_dependencies([self.final_context_state.cell_state[1]]):
            context_updates = []
            for s in range(NUM_SPEAKERS):
                update_context = tf.assign(
                    self.contexts[s],
                    tf.where(
                        tf.equal(self.is_first_utts, False),
                        tf.where(
                            tf.equal(self.speakers, s),
                            self.final_context_state.cell_state,
                            self.contexts[s],
                        ),
                        self._get_decoder_cell().zero_state(self.hparams.batch_size, dtype=tf.float32)
                    )
                )
                context_updates.append(update_context)

            self._extra_ops = tf.group(context_updates)

        return ret