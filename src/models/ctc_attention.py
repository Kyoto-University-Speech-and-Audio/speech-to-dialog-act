#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base import BaseModel

import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from six.moves import xrange as range

from csp.utils import ops_utils, model_utils

NUM_UNITS = 320
LAMBDA = 0.5

# Hyper-parameters
num_epochs = 10000
num_hidden = 50
num_encoder_layers = 3
num_decoder_layers = 3
initial_learning_rate = 1e-3
momentum = 0.9

max_gradient_norm = 5.0


class CTCAttentionModel(BaseModel):
    def _build_graph(self):
        if False:
            self.TGT_SOS_INDEX = self.hparams.vocab_size
            self.TGT_EOS_INDEX = self.hparams.vocab_size + 1
            self.vocab_size = self.hparams.vocab_size + 2
            # Add sos and eos
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.target_labels = tf.concat([
                    tf.fill([self.batch_size, 1], self.TGT_SOS_INDEX),
                    self.target_labels,
                    tf.fill([self.batch_size, 1], self.TGT_EOS_INDEX)
                ], 1)
                self.target_seq_len = tf.add(2, self.target_seq_len)
        else:
            self.TGT_EOS_INDEX = 0
            self.TGT_SOS_INDEX = 1
            self.vocab_size = self.hparams.vocab_size

        self.target_labels_no_sos = tf.slice(self.target_labels, [0, 1],
                                             [tf.shape(self.target_labels)[0], tf.shape(self.target_labels)[1] - 1])

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.targets = tf.one_hot(self.target_labels, depth=self.vocab_size)

        # Projection
        self.output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection")

        encoder_outputs, encoder_state = self._build_encoder()
        attention_logits, self.sample_id, final_context_state = self._build_decoder(encoder_outputs, encoder_state)
        ctc_logits = self._build_ctc(encoder_outputs)

        # Loss
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.loss = self.compute_loss(attention_logits, ctc_logits)

    def compute_loss(self, attention_logits, ctc_logits):
        # Attention loss
        max_time = tf.shape(attention_logits)[1]

        target_labels = ops_utils.pad_tensor(
            self.target_labels_no_sos,
            tf.shape(attention_logits)[1],
            self.TGT_EOS_INDEX, axis=1)

        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels,
            logits=attention_logits)

        target_weights = tf.sequence_mask(self.target_seq_len, max_time, dtype=attention_logits.dtype)

        attention_loss = tf.reduce_mean(cross_ent * target_weights)

        # CTC loss
        sparse_targets = ops_utils.sparse_tensor(self.target_labels, padding_value=-1)
        ctc_loss = tf.nn.ctc_loss(sparse_targets, ctc_logits, self.input_seq_len, ignore_longer_outputs_than_inputs=True)
        ctc_loss = tf.reduce_mean(ctc_loss)

        self.compute_ler(attention_logits, target_labels)

        return ctc_loss * LAMBDA + attention_loss * (1 - LAMBDA)

    def compute_ler(self, logits, target_labels):
        self.sample_words = tf.argmax(logits, axis=-1)
        # letter error rate
        self.ler = tf.reduce_mean(tf.edit_distance(
            ops_utils.sparse_tensor(self.sample_words),
            ops_utils.sparse_tensor(tf.cast(target_labels, tf.int64))))

    def _build_encoder(self):
        cells_fw = [model_utils.single_cell("lstm", NUM_UNITS, self.mode) for _ in range(num_encoder_layers)]
        cells_bw = [model_utils.single_cell("lstm", NUM_UNITS, self.mode) for _ in range(num_encoder_layers)]
        outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw,
            self.inputs, sequence_length=self.input_seq_len,
            dtype=tf.float32)

        return outputs, tf.concat([output_states_fw, output_states_bw], -1)

    def _build_ctc(self, encoder_outputs):
        logits = tf.layers.dense(encoder_outputs, self.vocab_size)
        logits = tf.transpose(logits, (1, 0, 2))
        return logits

    def _build_decoder(self, encoder_outputs, encoder_final_state):
        with tf.variable_scope('decoder') as decoder_scope:
            # cells = [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_decoder_layers)]
            # decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
            decoder_cell = model_utils.single_cell("lstm", NUM_UNITS * 2, self.mode)

            self.embedding_decoder = tf.diag(tf.ones(self.vocab_size))

            attention_mechanism = CustomAttention(
                NUM_UNITS * 2,
                encoder_outputs,
                memory_sequence_length=self.input_seq_len
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                alignment_history=False,
                attention_layer_size=NUM_UNITS * 2,
            )

            decoder_initial_state = decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)
            # decoder_initial_state = decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)\
            #    .clone(cell_state=encoder_final_state)
            # decoder_initial_state = encoder_final_state
            decoder_emb_layer = tf.layers.Dense(NUM_UNITS * 2)

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                decoder_emb_inp = decoder_emb_layer(self.targets)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.target_seq_len)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.vocab_size)),
                    start_tokens=tf.fill([self.batch_size], self.TGT_SOS_INDEX),
                    end_token=self.TGT_EOS_INDEX
                )

            if self.mode != tf.estimator.ModeKeys.PREDICT:
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_initial_state,
                )

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, swap_memory=True,
                    scope=decoder_scope,
                    maximum_iterations=100)

                sample_id = outputs.sample_id

                logits = self.output_layer(outputs.rnn_output)

            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_initial_state,
                    output_layer=self.output_layer
                )

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    swap_memory=True,
                    maximum_iterations=100,
                    scope=decoder_scope)

                logits = None
                sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def train(self, sess):
        _, loss, self.summary, targets, ler, sample_id = sess.run([
            self.update,
            self.loss,
            self.train_summary,
            self.targets,
            self.ler,
            self.sample_id,
        ])
        return loss, ler


    def eval(self, sess):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        target_labels, test_loss, sample_words, ler = sess.run([
            self.target_labels,
            self.loss,
            self.sample_words,
            self.ler
        ])
        return target_labels[:, 1:-1], test_loss, ler, sample_words

    def infer(self, sess):
        sample_id = sess.run([
            self.sample_id
        ])

        return sample_id


from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism


class CustomAttention(_BaseAttentionMechanism):
    def __init__(self, NUM_UNITS, memory, memory_sequence_length):
        super(CustomAttention, self).__init__(
            query_layer=layers_core.Dense(NUM_UNITS, name="query_layer", use_bias=False, dtype=tf.float32),
            memory_layer=layers_core.Dense(NUM_UNITS, name="memory_layer", use_bias=False, dtype=tf.float32),
            memory=memory,
            probability_fn=lambda score, _: tf.nn.softmax(score),
            memory_sequence_length=memory_sequence_length,
            score_mask_value=None,
            name="CustomAttention"
        )

        self._num_units = NUM_UNITS

    def __call__(self, query, state):
        with tf.variable_scope(None, "custom_attention", [query]):
            processed_query = self.query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1)
            b = tf.get_variable("attention_b", [self._num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
            v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(processed_query + self._keys + b), [2])
            alignments = tf.nn.softmax(score)
            next_state = alignments
            return alignments, next_state
