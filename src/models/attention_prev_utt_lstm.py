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
#from .attentions.attention_wrapper import AttentionWrapper
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper as BaseHelper

NUM_SPEAKERS = 2

class FixedHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        result.set_shape([3])
        return result


class AttentionModel(BaseAttentionModel):
    def __init__(self,
                 feed_initial_state=False):
        super().__init__(
            # attention_wrapper_fn=self._attention_wrapper,
            train_decode_fn=self._train_decode_fn,
            eval_decode_fn=self._eval_decode_fn,
            greedy_embedding_helper_fn=lambda embedding, start_tokens, end_token:
                FixedHelper(embedding, start_tokens, end_token)
        )
        self._feed_initial_state = feed_initial_state

    def __call__(self, hparams, mode, iterator, **kwargs):
        if mode == tf.estimator.ModeKeys.EVAL:
            hparams = tf.contrib.training.HParams(**hparams.values())
            hparams.batch_size = 3
            print(hparams)
        super().__call__(hparams, mode, iterator)

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
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {var.op.name: var for var in saver_variables}

        del var_list["Variable"]
        del var_list["Variable_1"]

        for s in range(NUM_SPEAKERS): del var_list["context_speaker_%d" % s]
        for s in range(NUM_SPEAKERS):
            if "context_speaker_%d_1" % s in var_list: del var_list["context_speaker_%d_1" % s]

        loaded_kernel = tf.get_variable("loaded_kernel", shape=[1920, 2560], initializer=tf.zeros_initializer)
        var_list["decoder/attention_wrapper/basic_lstm_cell/kernel"] = loaded_kernel
        if flags.mode == "train":
            loaded_kernel_adam = tf.get_variable("loaded_kernel_adam", shape=[1920, 2560], initializer=tf.zeros_initializer)
            loaded_kernel_adam1 = tf.get_variable("loaded_kernel_adam1", shape=[1920, 2560], initializer=tf.zeros_initializer)

            var_list["decoder/attention_wrapper/basic_lstm_cell/kernel/Adam"] = loaded_kernel_adam
            var_list["decoder/attention_wrapper/basic_lstm_cell/kernel/Adam_1"] = loaded_kernel_adam1

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

        var_list = {var.op.name: var for var in saver_variables}
        sess.run([
            tf.assign(var_list["decoder/attention_wrapper/basic_lstm_cell/kernel"],
                      tf.concat([loaded_kernel[:640], tf.zeros([640, 2560]), loaded_kernel[640:]], axis=0)),
            tf.assign(var_list["decoder/attention_wrapper/basic_lstm_cell/kernel/Adam"],
                      tf.concat([loaded_kernel_adam[:640], tf.zeros([640, 2560]), loaded_kernel_adam[640:]], axis=0)) if flags.mode == "train" else tf.no_op(),
            tf.assign(var_list["decoder/attention_wrapper/basic_lstm_cell/kernel/Adam_1"],
                      tf.concat([loaded_kernel_adam1[:640], tf.zeros([640, 2560]), loaded_kernel_adam1[640:]], axis=0)) if flags.mode == "train" else tf.no_op()
        ])

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

        #self.full_context = tf.concat(self.contexts, axis=-1)

        ret = super()._build_graph()

        print("final", self.final_context_state.cell_state)
        with tf.control_dependencies([self.final_context_state.cell_state[1]]):
            context_updates = []
            for s in range(NUM_SPEAKERS):
                for state_id in range(2):
                    update_context = tf.assign(
                        self.contexts[s][state_id],
                        tf.where(
                            tf.equal(self.is_first_utts, False),
                            tf.where(
                                tf.equal(self.speakers, s),
                                self.final_context_state.cell_state[state_id],
                                self.contexts[s][state_id],
                            ),
                            self._get_decoder_cell().zero_state(self.hparams.batch_size, dtype=tf.float32)[state_id]
                        )
                    )
                    context_updates.append(update_context)

            self._extra_ops = tf.group(context_updates)

        return ret

    def _build_context(self, decoder_cell, encoder_outputs):
        context = tf.where(
            tf.equal(self.speakers, 0),
            self.contexts[0][1],
            self.contexts[1][1]
        )
        initial_state = [None] * NUM_SPEAKERS
        for s in range(NUM_SPEAKERS):
            initial_state[s] = tf.where(
                tf.equal(self.speakers, 0),
                self.contexts[1][s],
                self.contexts[0][s]
            )
        initial_state = LSTMStateTuple(initial_state[0], initial_state[1])
        initial_state = self._get_attention_cell(decoder_cell, encoder_outputs) \
            .zero_state(self.hparams.batch_size, tf.float32) \
            .clone(cell_state=initial_state)
        return context, initial_state


    def _train_decode_fn(self, decoder_inputs, target_seq_len, initial_state, encoder_outputs, decoder_cell, scope):
        context, initial_state = self._build_context(decoder_cell, encoder_outputs)
        return super()._train_decode_fn_default(
            decoder_inputs, target_seq_len,
            initial_state if self._feed_initial_state else None, encoder_outputs, decoder_cell, scope, context)

    def _eval_decode_fn(self, initial_state, encoder_outputs, decoder_cell, scope):
        context, initial_state = self._build_context(decoder_cell, encoder_outputs)
        return super()._eval_decode_fn_default(
            initial_state if self._feed_initial_state else None,
            encoder_outputs, decoder_cell, scope, context)

    @classmethod
    def trainable_variables(cls):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
