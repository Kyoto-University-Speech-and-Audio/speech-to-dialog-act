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

class AttentionModel(BaseModel):
    def __init__(self,
                 attention_fn=None,
                 beam_search_decoder_cls=BeamSearchDecoder,
                 train_decode_fn=None,
                 eval_decode_fn=None,
                 attention_wrapper_fn=tf.contrib.seq2seq.AttentionWrapper,
                 greedy_embedding_helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper,
                 force_alignment_history=True):
        super().__init__()
        self._attention_fn = attention_fn or self._attention_fn_default
        self._beam_search_decoder_cls = beam_search_decoder_cls
        self._train_decode_fn = train_decode_fn or self._train_decode_fn_default
        self._eval_decode_fn = eval_decode_fn or self._eval_decode_fn_default
        self._attention_wrapper_fn = attention_wrapper_fn
        self._attention_cell = None
        self._greedy_embedding_helper_fn = greedy_embedding_helper_fn
        self._alignment_history = force_alignment_history

    def __call__(self, hparams, mode, batched_input, **kwargs):
        BaseModel.__call__(self, hparams, mode, batched_input, **kwargs)

        if self.infer_mode or self.eval_mode:
            attention_summary = self._get_attention_summary()
            if attention_summary is not None:
                self.summary = tf.summary.merge([attention_summary])
            else:
                self.summary = tf.no_op()

        return self
    
    def get_ground_truth_label_placeholder(self): return [self.targets]
    def get_predicted_label_placeholder(self): return [self.sample_id]
    
    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len), (self.targets, self.target_seq_len)) = \
                self.iterator.get_next()
        else:
            self.input_filenames, self.inputs, self.input_seq_len = self.iterator.get_next()

    def _build_graph(self):
        if not self.infer_mode:
            self.one_hot_targets = tf.one_hot(self.targets, depth=self.hparams.num_classes)
            # remove <sos> in target labels to feed into output
            target_labels = tf.slice(self.targets, [0, 1],
                                     [self.batch_size, tf.shape(self.targets)[1] - 1])
            target_labels = tf.concat([
                target_labels,
                tf.fill([self.batch_size, 1], self.hparams.eos_index)
            ], 1)
            self.target_labels = target_labels

        # Projection layer
        self.output_layer = layers_core.Dense(self.hparams.num_classes, use_bias=False, name="output_projection")

        encoder_outputs, encoder_state = self._build_encoder()
        self.encoder_outputs = encoder_outputs

        logits, self.sample_id, self.final_context_state = \
            self._build_decoder(encoder_outputs, encoder_state)
        self.logits = logits

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
                cells = [model_utils.single_cell("lstm", self.hparams.encoder_num_units, self.mode) for _ in
                         range(self.hparams.num_encoder_layers)]
                cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, sequence_length=self.input_seq_len, dtype=tf.float32)
                print(state)
                return outputs, state

    def _get_attention_cell(self, decoder_cell, encoder_outputs):
        if self._attention_cell is not None: return self._attention_cell
        attention_mechanism = self._attention_fn(encoder_outputs)

        cell = self._attention_wrapper_fn(
            decoder_cell, attention_mechanism,
            attention_layer_size=self.hparams.attention_layer_size,
            alignment_history=self._alignment_history,
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
                _output_layer=self.output_layer
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
            scale=self.hparams.attention_energy_scale,
            location_conv_size=(10, self.hparams.location_attention_width)
        )

    def _get_decoder_cell(self):
        if self.hparams.num_decoder_layers == 1:
            self._decoder_cell = model_utils.single_cell("lstm", self.hparams.decoder_num_units, self.mode)
        else:
            cells = [model_utils.single_cell("lstm", self.hparams.decoder_num_units, self.mode) for _ in
                     range(self.hparams.num_decoder_layers)]
            self._decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            #outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, sequence_length=self.input_seq_len, dtype=tf.float32)

        return self._decoder_cell

    def _build_decoder(self, encoder_outputs, encoder_final_state):
        with tf.variable_scope('decoder') as decoder_scope:
            self.embedding_decoder = tf.diag(tf.ones(self.hparams.num_classes))

            # decoder_initial_state = decoder_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)\
            #    .clone(cell_state=encoder_final_state)
            # decoder_initial_state = encoder_final_state
            decoder_cell = self._get_decoder_cell()
            self.decoder_emb_layer = tf.layers.Dense(self.hparams.decoder_num_units, name="decoder_emb_layer")

            if self.train_mode:
                decoder_emb_inp = self.decoder_emb_layer(self.one_hot_targets)

                if self.hparams.use_encoder_final_state:
                    if encoder_final_state.shape[-1] == self.hparams.decoder_num_units:
                        initial_state = tf.contrib.rnn.LSTMStateTuple(encoder_final_state[0], encoder_final_state[1])
                    else:
                        initial_state = tf.layers.dense(encoder_final_state, self.hparams.decoder_num_units)
                else: initial_state = None

                outputs, final_context_state, self.final_sequence_lengths = self._train_decode_fn(
                    decoder_emb_inp,
                    self.target_seq_len,
                    initial_state,
                    encoder_outputs,
                    decoder_cell,
                    scope=decoder_scope
                )

                self.decoder_outputs = outputs.rnn_output
                #self.decoder_states = outputs.
                logits = self.output_layer(outputs.rnn_output)
                sample_ids = tf.argmax(logits, axis=-1)
            else:
                outputs, final_context_state, self.final_sequence_lengths = self._eval_decode_fn(
                    initial_state=None,
                    encoder_outputs=encoder_outputs,
                    decoder_cell=decoder_cell,
                    scope=decoder_scope)

                if self.hparams.beam_width > 0:
                    sample_ids = outputs.predicted_ids[:, :, 0]
                    logits = tf.one_hot(sample_ids, depth=self.hparams.num_classes)
                else:
                    self.decoder_outputs = outputs.rnn_output
                    sample_ids = outputs.sample_id
                    logits = outputs.rnn_output

        return logits, sample_ids, final_context_state

    def get_extra_ops(self):
        return []
        return [self.decoder_outputs]
        #return tf.no_op()
        attention = tf.transpose(self.final_context_state.alignment_history.stack(), [1, 0, 2]) # [batch_size * target_max_len * T]
        attention = tf.expand_dims(attention, -1)
        encoder_outputs = tf.expand_dims(self.encoder_outputs, 1)
        da_inputs = tf.reduce_mean(
            tf.multiply(
                attention,
                encoder_outputs,
            ),
            axis=2
        )
        return [da_inputs]
        #return tf.no_op()

    def eval(self, sess):
        return None
        input_filenames, target_labels, input_seq_len, loss, sample_ids, summary, images = sess.run([
            self.input_filenames,
            self.targets,
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
    
    def trainable_variables(self):
        trainable_vars = tf.trainable_variables()
        if self.hparams.freeze_encoder:
            return list(filter(lambda var: var.op.name[:7] != "encoder", trainable_vars))
        else:
            return trainable_vars

    def _get_attention_summary(self):
        return None
        self.attention_images = tf.no_op()
        if self.hparams.beam_width > 0: return None
        attention_images = self.final_context_state.alignment_history.stack()  # max_len * batch_size * T
        attention_images = tf.transpose(attention_images, [1, 0, 2])
        self.attention_images = attention_images

        return None
        inferred_labels = self.sample_id[0, :]
        indices = tf.where(tf.not_equal(inferred_labels, tf.constant(1, inferred_labels.dtype)))
        attention_images = attention_images[:self.input_seq_len[0], :tf.shape(indices)[0]]
        attention_images = tf.expand_dims(attention_images, 0)
        return None
        attention_images = tf.expand_dims(tf.transpose(attention_images, [0, 2, 1]),
                                          -1)  # batch_size * max_len * # T * 1
        attention_images = 1 - attention_images
        attention_images *= 255
        attention_summary = tf.summary.image("attention_images", attention_images, max_outputs=1)
        return attention_summary
    
    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {}
        for var in saver_variables:
            if var.op.name[:7] == "encoder":
                var_list[var.op.name] = var
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    def output_result(ground_truth_labels, predicted_labels, extra_ops):
        target_ids = ground_truth_labels[0]
        sample_ids = predicted_labels[0]
        atts = extra_ops[0]
        with open(self.hparams.result_output_file, "a") as f:
            for ids1, ids2, att in zip(target_ids, sample_ids,
                    atts):
                _ids1 = [str(id) for id in ids1 if id < self.hparams.num_classes - 2]
                _ids2 = [str(id) for id in ids2 if id < self.hparams.num_classes - 2]
                fn = "%s/%d.npy" % (self.hparams.result_output_folder, self._eval_count)
                f.write('\t'.join([
                    #filename.decode(),
                    ' '.join(_ids1),
                    ' '.join(_ids2),
                    #' '.join(self._batched_input_test.decode(ids2)),
                    fn
                ]) + '\n')
                att = att[:len(_ids2), :]
                np.save(fn, att)
