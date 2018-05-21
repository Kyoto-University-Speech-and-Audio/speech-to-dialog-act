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

# Hyper-parameters
num_epochs = 10000
num_hidden = 50
num_encoder_layers = 3
num_decoder_layers = 3
initial_learning_rate = 1e-3
momentum = 0.9

BEAM_WIDTH = 0
LENGTH_PENALTY_WEIGHT = 0.0

max_gradient_norm = 5.0

class AttentionModel(BaseModel):
    def __init__(self, hparams, mode, iterator):
        self.num_units = 2 * NUM_UNITS
        BaseModel.__init__(self, hparams, mode, iterator)

    def _build_graph(self):
        if False:
            self.TGT_SOS_INDEX = self.hparams.num_classes
            self.TGT_EOS_INDEX = self.hparams.num_classes + 1
            self.num_classes = self.hparams.num_classes + 2
            # Add sos and eos
            if not self.infer_mode:
                self.target_labels = tf.concat([
                    tf.fill([self.batch_size, 1], self.TGT_SOS_INDEX),
                    self.target_labels,
                    tf.fill([self.batch_size, 1], self.TGT_EOS_INDEX)
                ], 1)
                self.target_seq_len = tf.add(2, self.target_seq_len)
        else:
            self.TGT_EOS_INDEX = 0
            self.TGT_SOS_INDEX = 1
            self.num_classes = self.hparams.num_classes

        if not self.infer_mode:
            self.targets = tf.one_hot(self.target_labels, depth=self.num_classes)

        # Projection
        self.output_layer = layers_core.Dense(self.num_classes, use_bias=False, name="output_projection")
        
        # Encoder
        encoder_outputs, encoder_state = self._build_encoder()
        
        # Decoder
        logits, self.sample_id, self.final_context_state = \
            self._build_decoder(encoder_outputs, encoder_state)

        self.max_time = tf.shape(self.sample_id)[1]

        if not self.infer_mode:
            # Loss
            self._compute_loss(logits)

    def _compute_loss(self, logits):
        target_labels = tf.slice(self.target_labels, [0, 1],
                                 [tf.shape(self.target_labels)[0], tf.shape(self.target_labels)[1] - 1])
        target_labels = tf.concat([
            target_labels,
            tf.fill([self.batch_size, self.max_time - tf.shape(target_labels)[1]], self.TGT_EOS_INDEX)
        ], 1)

        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_labels,
            logits=logits)

        target_weights = tf.sequence_mask(self.target_seq_len, self.max_time, dtype=logits.dtype)

        self.loss = tf.reduce_mean(cross_ent * target_weights)

        self.ler = self._compute_ler(self.sample_id, target_labels)


    # label error rate
    def _compute_ler(self, sample_ids, target_labels):
        return tf.reduce_mean(tf.edit_distance(
            ops_utils.sparse_tensor(tf.cast(sample_ids, tf.int64)),
            ops_utils.sparse_tensor(tf.cast(target_labels, tf.int64))))

    def _build_encoder(self):
        cells_fw = [model_utils.single_cell("lstm", self.num_units // 2, self.mode) for _ in range(num_encoder_layers)]
        cells_bw = [model_utils.single_cell("lstm", self.num_units // 2, self.mode) for _ in range(num_encoder_layers)]
        outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw,
            self.inputs, sequence_length=self.input_seq_len,
            dtype=tf.float32)

        return outputs, tf.concat([output_states_fw, output_states_bw], -1)

    def _build_decoder(self, encoder_outputs, encoder_final_state):
        with tf.variable_scope('decoder') as decoder_scope:
            # cells = [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_decoder_layers)]
            # decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)
            decoder_cell = model_utils.single_cell("lstm", self.num_units, self.mode)

            self.embedding_decoder = tf.diag(tf.ones(self.num_classes))

            if not self.train_mode and BEAM_WIDTH > 0:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(
                    encoder_outputs, multiplier=BEAM_WIDTH)
                self.input_seq_len = tf.contrib.seq2seq.tile_batch(
                    self.input_seq_len, multiplier=BEAM_WIDTH)
                batch_size = self.batch_size * BEAM_WIDTH
            else:
                batch_size = self.batch_size

            attention_mechanism = CustomAttention(
                self.num_units,
                encoder_outputs,
                memory_sequence_length=self.input_seq_len
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=self.num_units,
                alignment_history=not self.train_mode and BEAM_WIDTH == 0,
                output_attention=True
            )

            decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)

            # decoder_initial_state = decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)\
            #    .clone(cell_state=encoder_final_state)
            # decoder_initial_state = encoder_final_state

            decoder_emb_layer = tf.layers.Dense(self.num_units)

            if self.train_mode:
                decoder_emb_inp = decoder_emb_layer(self.targets)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.target_seq_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_initial_state,
                )

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, swap_memory=True,
                    scope=decoder_scope,
                    maximum_iterations=100)

                logits = self.output_layer(outputs.rnn_output)
                sample_id = tf.argmax(logits, axis=-1)
            else:
                if BEAM_WIDTH > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        decoder_cell,
                        lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.num_classes)),
                        start_tokens=tf.fill([batch_size], self.TGT_SOS_INDEX),
                        end_token=self.TGT_EOS_INDEX,
                        initial_state=decoder_initial_state,
                        beam_width=BEAM_WIDTH,
                        output_layer=self.output_layer,
                        length_penalty_weight=LENGTH_PENALTY_WEIGHT,
                        )
                else: 
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.num_classes)),
                        start_tokens=tf.fill([batch_size], self.TGT_SOS_INDEX),
                        end_token=self.TGT_EOS_INDEX
                    )

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

                if BEAM_WIDTH > 0:
                    sample_id = outputs.predicted_ids[0]
                    logits = tf.one_hot(sample_id, depth=self.num_classes)
                else: 
                    sample_id = outputs.sample_id
                    logits = outputs.rnn_output

        return logits, sample_id, final_context_state

    def train(self, sess):
        _, loss, self.summary, ler, sample_id, global_step = sess.run([
            self.update,
            self.loss,
            self.train_summary,
            self.ler,
            self.sample_id,
            self.global_step
        ])
        return loss, ler, global_step

    def eval(self, sess):
        target_labels, test_loss, sample_ids, ler = sess.run([
            self.target_labels,
            self.loss,
            self.sample_id,
            self.ler
        ])
        return target_labels[:, 1:-1], test_loss, ler, sample_ids


    def infer(self, sess):
        sample_id = sess.run([
            self.sample_id
        ])

        return sample_id

    def _get_infer_summary(self, hparams):
        attention_images = self.final_context_state.alignment_history.stack()
        attention_images = tf.expand_dims(tf.transpose(attention_images, [1, 2, 0]), -1)
        attention_images *= 255
        attention_images = tf.summary.image("attention_images", attention_images)
        return attention_images


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

        self.num_units = num_units

    def __call__(self, query, state):
        with tf.variable_scope(None, "custom_attention", [query]):
            processed_query = self.query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1)
            b = tf.get_variable("attention_b", [self.num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
            v = tf.get_variable("attention_v", [self.num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(processed_query + self._keys + b), [2])
            alignments = tf.nn.softmax(score)
            next_state = alignments
            return alignments, next_state
