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

# Hyper-parameters
SAMPLING_TEMPERATURE = 0

max_gradient_norm = 5.0

class AttentionModel(BaseModel):
    def __init__(self, hparams, mode, iterator):
        self.num_units = 2 * hparams.num_units
        BaseModel.__init__(self, hparams, mode, iterator)

        if self.infer_mode or self.eval_mode:
            attention_summary = self._get_attention_summary(hparams)
            if attention_summary is not None:
                self.summary = tf.summary.merge([attention_summary])
            else: self.summary = tf.no_op()

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
            self.num_classes = self.hparams.num_classes

        if not self.infer_mode:
            self.targets = tf.one_hot(self.target_labels, depth=self.num_classes)
            # remove <sos> in target labels to feed into output
            target_labels = tf.slice(self.target_labels, [0, 1],
                                     [self.batch_size, tf.shape(self.target_labels)[1] - 1])
            target_labels = tf.concat([
                target_labels,
                tf.fill([self.batch_size, 1], self.hparams.eos_index)
            ], 1)

        # Projection layer
        self.output_layer = layers_core.Dense(self.num_classes, use_bias=False, name="output_projection")
        
        encoder_outputs, encoder_state = self._build_encoder()
        
        logits, self.sample_id, self.final_context_state = \
            self._build_decoder(encoder_outputs, encoder_state)

        self.max_time = tf.shape(self.sample_id)[1]

        if self.train_mode or self.eval_mode:
            loss = self._compute_loss(logits, target_labels)
        else:
            loss = None
        return loss

    def _compute_loss(self, logits, target_labels):
        if self.eval_mode:
            max_size = tf.maximum(tf.shape(target_labels)[1], tf.shape(logits)[1])
            _target_labels = tf.pad(target_labels, [[0, 0], [0, max_size - tf.shape(target_labels)[1]]])
            _logits = tf.pad(logits, [[0, 0], [0, max_size - tf.shape(logits)[1]], [0, 0]])
            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=_target_labels,
                logits=_logits)
            # target_weights = tf.sequence_mask(self.target_seq_len, self.max_time, dtype=logits.dtype)
            return tf.reduce_mean(cross_ent)
        else:
            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_labels,
                logits=logits)
            target_weights = tf.sequence_mask(self.target_seq_len, self.max_time, dtype=logits.dtype)
            return tf.reduce_mean(cross_ent * target_weights)

    def _build_encoder(self):
        cells_fw = [model_utils.single_cell("lstm", self.num_units // 2, self.mode) for _ in range(self.hparams.num_encoder_layers)]
        cells_bw = [model_utils.single_cell("lstm", self.num_units // 2, self.mode) for _ in range(self.hparams.num_encoder_layers)]
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

            if not self.train_mode and self.hparams.beam_width > 0:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(
                    encoder_outputs, multiplier=self.hparams.beam_width)
                self.input_seq_len = tf.contrib.seq2seq.tile_batch(
                    self.input_seq_len, multiplier=self.hparams.beam_width)
                batch_size = self.batch_size * self.hparams.beam_width
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
                alignment_history=not self.train_mode and self.hparams.beam_width == 0,
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
                    scope=decoder_scope)

                logits = self.output_layer(outputs.rnn_output)
                sample_ids = tf.argmax(logits, axis=-1)
            else:
                if self.hparams.beam_width > 0:
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        decoder_cell,
                        lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.num_classes)),
                        start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                        end_token=self.hparams.eos_index,
                        initial_state=decoder_initial_state,
                        beam_width=self.hparams.beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=self.hparams.length_penalty_weight,
                        )
                else:
                    if SAMPLING_TEMPERATURE > 0.0:
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.num_classes)),
                            start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                            end_token=self.hparams.eos_index,
                            softmax_temperature=SAMPLING_TEMPERATURE,
                            seed=1001
                        )
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            lambda ids: decoder_emb_layer(tf.one_hot(ids, depth=self.num_classes)),
                            start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                            end_token=self.hparams.eos_index
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

                if self.hparams.beam_width > 0:
                    sample_ids = outputs.predicted_ids[:, :, 0]
                    logits = tf.one_hot(sample_ids, depth=self.num_classes)
                else: 
                    sample_ids = outputs.sample_id
                    logits = outputs.rnn_output

        return logits, sample_ids, final_context_state

    def train(self, sess):
        _, loss, self.summary, sample_id, global_step = sess.run([
            self.update,
            
            self.loss,
            self.train_summary,
            self.sample_id,
            self.global_step
        ])
        return loss, global_step

    def eval(self, sess):
        target_labels, loss, sample_ids, summary = sess.run([
            self.target_labels,
            self.loss,
            self.sample_id,
            self.summary
        ])
        return target_labels[:, 1:-1], loss, sample_ids, summary


    def infer(self, sess):
        sample_ids, summary = sess.run([
            self.sample_id, self.summary
        ])

        return sample_ids, summary

    def _get_attention_summary(self, hparams):
        if self.hparams.beam_width > 0: return None
        attention_images = self.final_context_state.alignment_history.stack() # batch_size * T * max_len
        attention_images = tf.expand_dims(tf.transpose(attention_images, [1, 0, 2]), -1) # batch_size * max_len * # T * 1
        attention_images *= 255
        attention_summary = tf.summary.image("attention_images", attention_images, max_outputs=10)
        return attention_summary


from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism


class CustomAttention(_BaseAttentionMechanism):
    def __init__(self, num_units, 
            memory, memory_sequence_length,
            sharpening=False, smoothing=True,
            use_location_based_attention=True,
            location_conv_size=(10, 201)):

        if not smoothing:
            probability_fn = lambda score, _: tf.nn.softmax(score)
        else:
            def sigmoid_prob(score, _):
                sigmoids = tf.sigmoid(score)
                return sigmoids / tf.reduce_sum(sigmoids, axis=-1,
                        keep_dims=True)
            probability_fn = sigmoid_prob

        if not sharpening:
            memory_layer = layers_core.Dense(num_units, name="memory_layer", use_bias=False, dtype=tf.float32)
        #else:
            #memory_layer = layers_core.Dense()


        super(CustomAttention, self).__init__(
            query_layer=layers_core.Dense(num_units, name="query_layer", use_bias=False, dtype=tf.float32),
            memory_layer=memory_layer,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=None,
            probability_fn=probability_fn,
            name="CustomAttention"
        )

        self.num_units = num_units
        self.location_conv_size = location_conv_size
        self.use_location_based_attention = use_location_based_attention

    def __call__(self, query, state):
        with tf.variable_scope(None, "custom_attention", [query]):
            processed_query = self.query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1) # W * s_{i-1}

            
            if self.use_location_based_attention:
                #expanded_alignments = tf.expand_dims(tf.expand_dims(state, axis=-1), axis=-1)
                #f = tf.layers.conv2d(expanded_alignments, self.num_units,
                #        self.location_conv_size, padding='same',
                #        use_bias=False, name='location_conv')
                #f = tf.squeeze(f, [2])
                #processed_location = tf.layers.dense(f, self.num_units,
                #        use_bias=False, name='location_layer')
                expanded_alignments = tf.expand_dims(state, axis=-1)
                f = tf.layers.conv1d(expanded_alignments,
                        self.location_conv_size[1],
                        [self.location_conv_size[0]],
                        padding='same', use_bias=False, name='location_conv')
                processed_location = tf.layers.dense(f,
                        self.num_units,
                        use_bias=False, name='location_layer') # U * f_{i, j}
            else: processed_location = tf.no_op()

            #b = tf.get_variable("attention_b", [self.num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
            v = tf.get_variable("attention_v", [self.num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(processed_query +
                processed_location + self.keys), [2])

            alignments = self._probability_fn(score, state)
            
            next_state = alignments
            return alignments, next_state
