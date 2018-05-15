#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from six.moves import xrange as range

num_units = 50 # Number of units in the LSTM cell

# Hyper-parameters
num_epochs = 10000
num_hidden = 50
num_encoder_layers = 1
num_decoder_layers = 1
initial_learning_rate = 1e-2
momentum = 0.9

max_gradient_norm = 5.0

class AttentionModel():
    def __init__(self, hparams, mode, iterator):
        self.hparams = hparams
        self.iterator = iterator
        self.mode = mode

        self.TGT_SOS_INDEX = hparams.num_classes
        self.TGT_EOS_INDEX = hparams.num_classes + 1

        num_classes = hparams.num_classes + 2

        ((self.inputs, self.input_seq_len), (self.target_labels, self.target_seq_len)) = \
            iterator.get_next()

        self.target_labels = tf.concat([
            tf.fill([hparams.batch_size, 1], self.TGT_SOS_INDEX),
            self.target_labels,
            tf.fill([hparams.batch_size, 1], self.TGT_EOS_INDEX)
        ], 1)

        self.target_seq_len = tf.add(2, self.target_seq_len)

        self.targets = tf.one_hot(self.target_labels, depth=num_classes)

        # Projection
        self.output_layer = layers_core.Dense(num_classes, use_bias=False, name="output_projection")

        self.logits, loss, final_context_state, self.sample_id = self.build_graph()
        self.loss = loss

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(initial_learning_rate)

            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        self.saver = tf.train.Saver(tf.global_variables())

    def build_graph(self):
        encoder_outputs, encoder_state = self.build_encoder()
        logits, sample_id, final_context_state = self.build_decoder(encoder_outputs, encoder_state)

        # Loss
        max_time = self.targets.shape[1]
        crossent = tf.losses.softmax_cross_entropy(self.targets, logits)

        loss = tf.reduce_sum(crossent) / tf.to_float(self.hparams.batch_size)

        tf.summary.scalar("loss", loss)
        self.merged_summaries = tf.summary.merge_all()

        return logits, loss, final_context_state, sample_id

    def build_encoder(self):
        cells = [tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0) for _ in range(num_encoder_layers)]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

        encoder_outputs, encoder_state = \
            tf.nn.dynamic_rnn(encoder_cell, self.inputs, sequence_length=self.input_seq_len, dtype=tf.float32)
        return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_final_state):
        with tf.variable_scope('decoder') as decoder_scope:
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0) for _ in range(num_decoder_layers)]
            decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

            decoder_emb_inp = tf.layers.dense(self.targets, num_units)

            self.embedding_decoder = tf.diag(tf.ones(self.hparams.num_classes))

            attention_mechanism = CustomAttention(
                num_units,
                encoder_outputs,
                memory_sequence_length=self.input_seq_len
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=num_units,
            )

            if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.target_seq_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32),)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id

                logits = self.output_layer(outputs.rnn_output)

            elif self.mode == tf.estimator.ModeKeys.PREDICT:
                start_tokens = tf.fill([self.hparams.batch_size], self.TGT_SOS_INDEX)
                end_token = self.TGT_EOS_INDEX

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding_decoder,
                    start_tokens,
                    end_token
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,
                    output_layer=self.output_layer
                )

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    swap_memory=True,
                    maximum_iterations=100,
                    scope=decoder_scope)
                sample_id = outputs.sample_id
                logits = outputs.rnn_output
                self.sample_words = tf.argmax(sample_id, axis=1)

        return logits, sample_id, final_context_state

    def train(self, sess):
        _, loss, self.summary, logits, targets = sess.run([
            self.update_step,
            self.loss,
            self.merged_summaries,
            self.logits,
            self.targets
        ])
        return loss, 0


    def eval(self, sess):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        target_labels, test_cost, _, sample_id = sess.run([
            self.target_labels,
            self.loss,
            self.logits,
            self.sample_id
        ])
        return target_labels[:, 1:-1], test_cost, 0, sample_id[:, 1:-1]

from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism
class CustomAttention(_BaseAttentionMechanism):
    def __init__(self, num_units, memory, memory_sequence_length):
        super(CustomAttention, self).__init__(
            query_layer=layers_core.Dense(num_units, name="query_layer", use_bias=False, dtype=tf.float32),
            memory_layer=layers_core.Dense(num_units, name="memory_layer", use_bias=False, dtype=tf.float32),
            memory=memory,
            probability_fn=lambda score, _: tf.nn.softmax(score),
            memory_sequence_length=memory_sequence_length,
            score_mask_value=None,
            name="CustomAttention"
        )

        self._num_units = num_units

    def __call__(self, query, state):
        with tf.variable_scope(None, "custom_attention", [query]):
            processed_query = self.query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1)
            b = tf.get_variable("attention_b", [num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
            v = tf.get_variable("attention_v", [num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(processed_query + self._keys + b), [2])
            alignments = tf.nn.softmax(score)
            next_state = alignments
            return alignments, next_state