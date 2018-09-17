#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from .base_multi_gpus import MultiGPUBaseModel as BaseModel
from .base import BaseModel
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.layers import core as layers_core
from .attentions.location_based_attention import LocationBasedAttention

from six.moves import xrange as range

from ..utils import ops_utils, model_utils
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from .helpers.basic_context_decoder import BasicContextDecoder

# Hyper-parameters
SAMPLING_TEMPERATURE = 0

max_gradient_norm = 5.0

class Model(BaseModel):
    def __init__(self):
        super().__init__()

    def __call__(self, hparams, mode, iterator, **kwargs):
        BaseModel.__call__(self, hparams, mode, iterator, **kwargs)

        if self.infer_mode or self.eval_mode:
            self.summary = tf.no_op()

        return self

    def _build_graph(self):
        # Projection layer
        self.output_layer = layers_core.Dense(self.hparams.num_classes, use_bias=False, name="output_projection")

        encoder_outputs, encoder_state = self._build_encoder()

        logits, self.sample_id, self.final_context_state = \
            self._build_decoder(encoder_outputs, encoder_state)

        self.max_time = tf.shape(self.sample_id)[1]

        if self.train_mode or self.eval_mode:
            loss = self._compute_loss(logits, target_labels)
        else:
            loss = None

        final_utterance_state = tf.Variable(
                    self.encoder_cells[-1].zero_state(self.batch_size, tf.float32),
                    name="final_utterance_state",
                    trainable=False, dtype=tf.float32)
        with tf.control_dependencies([loss]):
            self.update_utterance_state = tf.assign(encoder_state, final_utterance_state)

        return loss

    def get_extra_ops(self):
        return self.update_utterance_state

    def _compute_loss(self, logits, target_labels):

            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_labels,
                logits=logits)
            target_weights = tf.sequence_mask(self.target_seq_len, self.max_time, dtype=logits.dtype)
            return tf.reduce_mean(cross_ent * target_weights)

    def _build_encoder(self):
        #if True:
        with tf.variable_scope('encoder') as encoder_scope:
            if self.hparams.encoder_type == 'pbilstm':
                cells_fw = [model_utils.single_cell("lstm", self.hparams.encoder_num_units // 2, self.mode) for _ in
                            range(self.hparams.num_encoder_layers)]
                cells_bw = [model_utils.single_cell("lstm", self.hparams.encoder_num_units // 2, self.mode) for _ in
                            range(self.hparams.num_encoder_layers)]

                prev_layer = self.inputs
                prev_seq_len = self.input_seq_len

                with tf.variable_scope("stack_p_bidirectional_rnn"):
                    state_fw = state_bw = None
                    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
                        initial_state_fw = None
                        initial_state_bw = None

                        size = tf.cast(tf.floor(tf.shape(prev_layer)[1] / 2), tf.int32)
                        prev_layer = prev_layer[:, :size * 2, :]
                        prev_layer = tf.reshape(prev_layer,
                                                [tf.shape(prev_layer)[0], size, prev_layer.shape[2] * 2])
                        prev_seq_len = tf.cast(tf.floor(prev_seq_len / 2), tf.int32)

                        with tf.variable_scope("cell_%d" % i):
                            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                cell_fw,
                                cell_bw,
                                prev_layer,
                                initial_state_fw=initial_state_fw,
                                initial_state_bw=initial_state_bw,
                                sequence_length=prev_seq_len,
                                dtype=tf.float32
                            )
                            # Concat the outputs to create the new input.
                            prev_layer = tf.concat(outputs, axis=2)
                        #states_fw.append(state_fw)
                        #states_bw.append(state_bw)

                return prev_layer, tf.concat([state_fw, state_bw], -1)
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
                self.encoder_cells = [model_utils.single_cell("lstm", self.hparams.encoder_num_units, self.mode) for _ in
                         range(self.hparams.num_encoder_layers)]
                cell = tf.nn.rnn_cell.MultiRNNCell(self.encoder_cells)
                outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, sequence_length=self.input_seq_len, dtype=tf.float32)
                return outputs, state

    def _get_attention_cell(self, decoder_cell, encoder_outputs):
        if self._attention_cell is not None: return self._attention_cell
        attention_mechanism = self._attention_fn(encoder_outputs)

        cell = self._attention_wrapper_fn(
            decoder_cell, attention_mechanism,
            attention_layer_size=self.hparams.attention_layer_size,
            alignment_history=not self.train_mode and self.hparams.beam_width == 0,
            output_attention=self.hparams.output_attention
        )
        self._attention_cell = cell
        return cell

    def _train_decode_fn_default(self, decoder_inputs, target_seq_len, initial_state, encoder_outputs, decoder_cell, scope, context=None):
        self._attention_cell = self._get_attention_cell(decoder_cell, encoder_outputs)

        if initial_state is None:
            initial_state = self._attention_cell.zero_state(self.batch_size, dtype=tf.float32)
        else:
            initial_state = self._attention_cell.zero_state(self.batch_size, dtype=tf.float32)\
                .clone(cell_state=initial_state)

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_seq_len)
        decoder = BasicContextDecoder(self._attention_cell, helper, initial_state, context=context)

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

        self._get_attention_cell(decoder_cell, encoder_outputs)

        if initial_state == None:
            initial_state = self._attention_cell.zero_state(batch_size, dtype=tf.float32)

        def embed_fn(ids):
            return self.decoder_emb_layer(tf.one_hot(ids, depth=self.hparams.num_classes))

        if self.hparams.beam_width > 0:
            decoder = self._beam_search_decoder_cls(
                self._attention_cell,
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
                self._attention_cell,
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

    def _attention_fn_default(self, encoder_outputs):
        return LocationBasedAttention(
            self.hparams.attention_num_units,
            encoder_outputs,
            memory_sequence_length=self.input_seq_len,
            scale=self.hparams.attention_energy_scale
        )

    def _get_decoder_cell(self):
        self._decoder_cell = model_utils.single_cell("lstm", self.hparams.decoder_num_units, self.mode)
        return self._decoder_cell

    def get_extra_ops(self):
        return tf.no_op()

    def eval(self, sess):
        input_filenames, target_labels, input_seq_len, loss, sample_ids, summary, images = sess.run([
            self.input_filenames,
            self.target_labels,
            self.input_seq_len,
            self.loss,
            self.sample_id,
            self.summary, self.attention_images
        ])

        if images is not None:
            for i in range(len(input_filenames)):
                id = os.path.basename(input_filenames[i].decode('utf-8')).split('.')[0]
                img = (1 - images[i]) * 255
                target_seq_len = np.where(sample_ids[i] == 1)[0][0] if len(np.where(sample_ids[i] == 1)[0]) > 0 else -1
                img = img[:target_seq_len, :input_seq_len[i]]
                fig = plt.figure(figsize=(20, 2))
                ax = fig.add_subplot(111)
                ax.imshow(img, aspect="auto")
                ax.set_title(id)
                ax.set_yticks(np.arange(0, target_seq_len, 5))
                fig.savefig(os.path.join(self.hparams.summaries_dir, "alignments", id + ".png"))
                plt.close()

        return input_filenames, target_labels[:, 1:-1], sample_ids, summary

    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            self.input_filenames, self.inputs, self.input_seq_len, self.dlg_turn_changed = \
                self._iterator.get_next()
        else:
            self.input_filenames, self.inputs, self.input_seq_len = self._iterator.get_next()