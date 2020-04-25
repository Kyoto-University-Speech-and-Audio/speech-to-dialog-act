#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .base_multi_gpus import MultiGPUBaseModel as BaseModel
from .base import BaseModel

import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from six.moves import xrange as range

from ..utils import model_utils
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from .helpers.basic_context_decoder import BasicContextDecoder

# Hyper-parameters
SAMPLING_TEMPERATURE = 0

max_gradient_norm = 5.0

class AttentionModel(BaseModel):
    def __init__(self,
                 beam_search_decoder_cls=BeamSearchDecoder,
                 train_decode_fn=None,
                 eval_decode_fn=None,
                 greedy_embedding_helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper):
        super().__init__()
        self._beam_search_decoder_cls = beam_search_decoder_cls
        self._train_decode_fn = train_decode_fn or self._train_decode_fn_default
        self._eval_decode_fn = eval_decode_fn or self._eval_decode_fn_default
        self._greedy_embedding_helper_fn = greedy_embedding_helper_fn

    def __call__(self, hparams, mode, iterator, **kwargs):
        BaseModel.__call__(self, hparams, mode, iterator, **kwargs)

        if self.infer_mode or self.eval_mode:
            attention_summary = None
            if attention_summary is not None:
                self.summary = tf.summary.merge([attention_summary])
            else:
                self.summary = tf.no_op()

        return self

    def _build_graph(self):
        if not self.infer_mode:
            self.targets = tf.one_hot(self.target_labels, depth=self.hparams.vocab_size)
            # remove <sos> in target labels to feed into output
            target_labels = tf.slice(self.target_labels, [0, 1],
                                     [self.batch_size, tf.shape(self.target_labels)[1] - 1])
            target_labels = tf.concat([
                target_labels,
                tf.fill([self.batch_size, 1], self.hparams.eos_index)
            ], 1)

        # Projection layer
        self.output_layer = layers_core.Dense(self.hparams.vocab_size, use_bias=False, name="output_projection")

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
        with tf.variable_scope('encoder') as encoder_scope:
            if self.hparams.encoder_type == 'bilstm':
                cells_fw = [model_utils.single_cell("lstm", self.hparams.encoder_num_units // 2, self.mode) for _ in
                            range(self.hparams.num_encoder_layers)]
                cells_bw = [model_utils.single_cell("lstm", self.hparams.encoder_num_units // 2, self.mode) for _ in
                            range(self.hparams.num_encoder_layers)]
                outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw, cells_bw,
                    self.inputs, sequence_length=self.input_seq_len,
                    dtype=tf.float32)
                return outputs, tf.concat([output_states_fw, output_states_bw], -1)
            elif self.hparams.encoder_type == 'lstm':
                cells = [model_utils.single_cell("lstm", self.hparams.encoder_num_units, self.mode) for _ in
                         range(self.hparams.num_encoder_layers)]
                cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, sequence_length=self.input_seq_len, dtype=tf.float32)
                print(state)
                return outputs, state

    def _train_decode_fn_default(self, decoder_inputs, target_seq_len, initial_state, encoder_outputs, decoder_cell, scope, context=None):
        if initial_state is None:
            initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_seq_len)
        decoder = BasicContextDecoder(decoder_cell, helper, initial_state, context=context)

        outputs, final_state, final_output_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder, swap_memory=True,
            scope=scope)

        return outputs, final_state, final_output_lengths

    def _eval_decode_fn_default(self, initial_state, encoder_outputs, decoder_cell, scope, context=None):
        if self.hparams.beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=self.hparams.beam_width)
            self.input_seq_len = tf.contrib.seq2seq.tile_batch(
                self.input_seq_len, multiplier=self.hparams.beam_width)
            batch_size = self.batch_size * self.hparams.beam_width
        else:
            batch_size = self.batch_size

        if initial_state == None:
            initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)

        def embed_fn(ids):
            return self.decoder_emb_layer(tf.one_hot(ids, depth=self.hparams.vocab_size))

        if self.hparams.beam_width > 0:
            decoder = self._beam_search_decoder_cls(
                decoder_cell,
                embed_fn,
                start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                end_token=self.hparams.eos_index,
                initial_state=initial_state,
                beam_width=self.hparams.beam_width,
                output_layer=self.output_layer,
                length_penalty_weight=self.hparams.length_penalty_weight,
            )
        else:
            if SAMPLING_TEMPERATURE > 0.0:
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    embed_fn,
                    start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                    end_token=self.hparams.eos_index,
                    softmax_temperature=SAMPLING_TEMPERATURE,
                    seed=1001
                )
            else:
                helper = self._greedy_embedding_helper_fn(
                    embed_fn,
                    start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
                    end_token=self.hparams.eos_index,
                )

            decoder = BasicContextDecoder(
                decoder_cell,
                helper,
                initial_state,
                context=context,
                output_layer=self.output_layer
            )

        outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True,
            maximum_iterations=100,
            scope=scope)

        return outputs, final_context_state, final_sequence_lengths

    def _get_decoder_cell(self):
        self._decoder_cell = model_utils.single_cell("lstm", self.hparams.decoder_num_units, self.mode)
        return self._decoder_cell

    def _build_decoder(self, encoder_outputs, encoder_final_state):
        with tf.variable_scope('decoder') as decoder_scope:
            self.embedding_decoder = tf.diag(tf.ones(self.hparams.vocab_size))

            # decoder_initial_state = decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)\
            #    .clone(cell_state=encoder_final_state)
            # decoder_initial_state = encoder_final_state
            decoder_cell = self._get_decoder_cell()
            self.decoder_emb_layer = tf.layers.Dense(self.hparams.decoder_num_units, name="decoder_emb_layer")

            if self.train_mode:
                decoder_emb_inp = self.decoder_emb_layer(self.targets)

                outputs, final_context_state, _ = self._train_decode_fn(
                    decoder_emb_inp,
                    self.target_seq_len,
                    None,
                    encoder_outputs,
                    decoder_cell,
                    scope=decoder_scope
                )

                logits = self.output_layer(outputs.rnn_output)
                sample_ids = tf.argmax(logits, axis=-1)
            else:
                outputs, final_context_state, final_sequence_lengths = self._eval_decode_fn(
                    initial_state=None,
                    encoder_outputs=encoder_outputs,
                    decoder_cell=decoder_cell,
                    scope=decoder_scope)

                if self.hparams.beam_width > 0:
                    sample_ids = outputs.predicted_ids[:, :, 0]
                    logits = tf.one_hot(sample_ids, depth=self.hparams.vocab_size)
                else:
                    sample_ids = outputs.sample_id
                    logits = outputs.rnn_output

        return logits, sample_ids, final_context_state

    def get_extra_ops(self):
        return tf.no_op()