#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from .base_multi_gpus import MultiGPUBaseModel as BaseModel
from models.base import BaseModel
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.layers import core as layers_core
from models.attentions.location_based_attention import LocationBasedAttention

from utils import ops_utils, model_utils, utils

max_gradient_norm = 5.0

class Model(BaseModel):
    default_params = {
        "use_sos_eos": True,
        "sos_index": None,
        "eos_index": None,
    }

    def __init__(self):
        super().__init__()

    def __call__(self, hparams, mode, iterator, **kwargs):
        BaseModel.__call__(self, hparams, mode, iterator, **kwargs)
        return self

    def _assign_input(self):
        if self.hparams.predicted_train_data is None:
            self.dlg_ids, (self.inputs, self.input_seq_len), \
            (self.targets, self.target_seq_len), \
            self.da_labels = self.iterator.get_next()
        else:
            self.dlg_ids, (self.inputs, self.input_seq_len), \
            (self.targets, self.target_seq_len), \
            (self.predicted_texts, self.predicted_text_seq_len), \
            self.acoustic_inputs, \
            self.da_labels = self.iterator.get_next()

    def get_ground_truth_label_placeholder(self): return [self.da_labels]
    def get_predicted_label_placeholder(self): return [self.predicted_da_labels]
    def get_ground_truth_label_len_placeholder(self): 
        return [tf.constant(1)]
    def get_predicted_label_len_placeholder(self): 
        return [tf.constant(1)]
    
    def get_decode_fns(self):
        return [
            lambda d: [str(d)]
        ]

    def _build_graph(self):
        if self.hparams.da_input == "acoustic_input":
            da_inputs = self.acoustic_inputs
            da_inputs = tf.layers.dense(self.acoustic_inputs, self.hparams.embedding_size)
            da_input_len = self.predicted_text_seq_len  
        elif self.hparams.da_input == "predicted_text":
            da_inputs = tf.one_hot(self.predicted_texts, self.hparams.vocab_size)
            da_inputs = tf.layers.dense(da_inputs,
                    self.hparams.embedding_size, name="dense")
            da_input_len = self.predicted_text_seq_len
        elif self.hparams.da_input == "ground_truth":
            da_inputs = tf.one_hot(self.targets, self.hparams.vocab_size)
            da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size,
                    name="dense")
            da_input_len = self.target_seq_len
        elif self.hparams.da_input == "combined":
            da_inputs1 = tf.layers.dense(self.acoustic_inputs,
                                         self.hparams.embedding_size / 2)
            da_input_len = self.predicted_text_seq_len
            da_inputs2 = tf.one_hot(self.predicted_texts, self.hparams.vocab_size)
            da_inputs2 = tf.layers.dense(da_inputs2,
                                         self.hparams.embedding_size / 2)
            da_inputs = tf.concat([da_inputs1, da_inputs2], -1)

        history_targets, history_target_seq_len, history_seq_len = self._build_history(
                da_inputs,
                da_input_len,
                dtype=tf.float32
        )

        history_inputs = self._build_word_encoder(
                history_targets,
                history_target_seq_len,
        )

        if self.hparams.num_utt_history != 0:
            encoded_history = self._build_utt_encoder(history_inputs, history_seq_len)
        else:
            encoded_history = tf.reduce_min(history_inputs, 1)

        loss, self.predicted_da_labels = self._get_loss(encoded_history)
        with tf.control_dependencies([loss]):
            self.update_prev_inputs = self._build_update_prev_inputs(da_inputs, da_input_len)

        return loss

    def _get_loss(self, encoded_history):
        # fully-connected layer
        self.da_logits = tf.layers.dense(
                encoded_history, 
                self.hparams.num_da_classes, name="logits"
        ) # [batch_size, num_da_classes]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.da_labels,
                logits=self.da_logits)

        return tf.reduce_mean(losses), tf.arg_max(self.da_logits, -1)

    def _build_update_prev_inputs(self, targets, target_seq_len):
        return tf.group([
            tf.assign(self.prev_dlg_ids, self.dlg_ids, validate_shape=False),
            tf.assign(self.prev_targets, targets, validate_shape=False),
            tf.assign(self.prev_target_seq_len, target_seq_len, validate_shape=False),
            #tf.assign(self.prev_speakers, self.speakers)
        ])


    def _build_utt_encoder(self, history_inputs, history_input_seq_len):
        # encoder at utterance-level
        history_encoder_outputs, history_encoder_state = tf.nn.dynamic_rnn(
                model_utils.single_cell("lstm",
                    self.hparams.utt_encoder_num_units, self.mode,
                    dropout=self.hparams.dropout), 
                history_inputs, 
                sequence_length=history_input_seq_len,
                dtype=tf.float32)
        return history_encoder_state[0]

    def _build_word_encoder(
            self, 
            history_targets, 
            history_target_seq_len,
            embedding_fn=None):
        """
        Args:
            history_targets: size [batch_size, history_len + 1, target_max_len]
            history_target_seq_len: size [batch_size, history_len + 1]

        Returns:
            history_inputs: size [batch_size, history_len + 1, num_hidden_units] 
                tensor of encoded utterances per batch
        """
        shape1 = tf.shape(history_targets)[0] # batch_size
        shape2 = tf.shape(history_targets)[1] # history_len + 1
        shape3 = tf.shape(history_targets)[2] # target_max_len

        # flatten history to feed into encoder at word-level
        utt_inputs = tf.reshape(history_targets, [shape1 * shape2, shape3, self.hparams.embedding_size])
        utt_input_seq_len = tf.reshape(history_target_seq_len, [shape1 * shape2])

        if embedding_fn is not None: utt_inputs = embedding_fn(utt_inputs)
        sent_encoder_outputs, sent_encoder_state = self._build_da_word_encoder(utt_inputs, utt_input_seq_len)
        
        # prepare input for encoder at utterance-level
        history_inputs = tf.reshape(sent_encoder_state[-1][0], [shape1, shape2, self.hparams.da_word_encoder_num_units])

        return history_inputs

    def _build_history(self, inputs, input_seq_len, rank=0, dtype=tf.int32):
        """
        Args:
            inputs: size [batch_size, target_max_len, ...]
            input_seq_len: size [batch_size]
        
        Returns:
            history_targets: size [batch_size, history_size + 1, target_max_len, ...]
            history_target_seq_len: size [batch_size, history_size + 1]
            history_seq_len: size [batch_size]
        """
        self.prev_targets = tf.get_variable("prev_targets", [0, 0, self.hparams.embedding_size], dtype=dtype, trainable=False)
        self.prev_target_seq_len = tf.get_variable("prev_target_seq_len", [0], tf.int32, trainable=False)
        self.prev_dlg_ids = tf.get_variable("prev_dlg_ids", [0], dtype=tf.string, 
                initializer=tf.constant_initializer('', tf.string),
                trainable=False)
        #self.prev_speakers = tf.placeholder(tf.bool, name="prev_speakers")

        # filter input from previous batch with same dialog id
        prev_same_dlg_targets = tf.boolean_mask(
                self.prev_targets,
                tf.equal(self.prev_dlg_ids, self.dlg_ids[0])
        )
        prev_same_dlg_target_seq_len = tf.boolean_mask(
                self.prev_target_seq_len,
                tf.equal(self.prev_dlg_ids, self.dlg_ids[0])
        )

        # concatenate previous batch with current batch
        full_targets = ops_utils.pad_and_concat(prev_same_dlg_targets, inputs, 0) # [num_utts, target_max_len, feature_size]
        full_target_seq_len = tf.concat([prev_same_dlg_target_seq_len, input_seq_len], 0) # [num_utts]
        
        # build history data
        def pad_history(t, i, rank):
            ret = t[tf.maximum(i - self.hparams.num_utt_history, 0):i + 1]
            return tf.pad(
                    ret, 
                    [[0, tf.maximum(0, self.hparams.num_utt_history - i)]] + [[0, 0]] * (rank - 1)
            )

        start_id = tf.shape(prev_same_dlg_targets)[0]
        end_id = tf.shape(full_targets)[0]
        history_targets = tf.map_fn(
                lambda i: pad_history(full_targets, i, 3), 
                tf.range(start_id, end_id),
                dtype=dtype
        ) # [batch_size, history_len + 1, max_utt_len, feature_size]
        history_target_seq_len = tf.map_fn(
                lambda i: pad_history(full_target_seq_len, i, 1),
                tf.range(start_id, end_id)
        ) # [batch_size, history_len + 1]
        history_seq_len = tf.map_fn(
                lambda i: tf.minimum(i + 1, self.hparams.num_utt_history + 1),
                tf.range(start_id, end_id)
        )
        return history_targets, history_target_seq_len, history_seq_len

    def _build_da_word_encoder(self, inputs, input_seq_len):
        #if True:
        with tf.variable_scope('encoder') as encoder_scope:
            if self.hparams.da_word_encoder_type == 'bilstm':
                cells_fw = [model_utils.single_cell("lstm",
                    self.hparams.da_word_encoder_num_units // 2, self.mode,
                    dropout=self.hparams.dropout) for _ in
                            range(self.hparams.num_da_word_encoder_layers)]
                cells_bw = [model_utils.single_cell("lstm",
                    self.hparams.da_word_encoder_num_units // 2, self.mode,
                    dropout=self.hparams.dropout) for _ in
                            range(self.hparams.num_da_word_encoder_layers)]
                outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw, cells_bw,
                    inputs, sequence_length=input_seq_len,
                    dtype=tf.float32)
                return outputs, tf.concat([output_states_fw, output_states_bw], -1)
            elif self.hparams.da_word_encoder_type == 'lstm':
                cells = [model_utils.single_cell("lstm", self.hparams.da_word_encoder_num_units, self.mode) for _ in
                         range(self.hparams.num_da_word_encoder_layers)]
                cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=input_seq_len, dtype=tf.float32)
                return outputs, state

    def get_extra_ops(self):
        return [self.update_prev_inputs, self.da_logits]

    def train(self, sess, extra_ops):
        ret = sess.run([
            self.inputs,
            self.targets,
            self.input_seq_len,
            self.get_extra_ops(),
        ] + extra_ops)

        inputs, target_labels, input_seq_len, extra_ret = ret[0], ret[1], ret[2], ret[-len(extra_ops):]

        if self.hparams.verbose:
            print("\nprocessed_inputs_count: %d, global_step: %d" % (self.processed_inputs_count, self.global_step))
            print("batch_size: %d, input_size: %d" % (len(inputs), len(inputs[0])))
            print("input_seq_len", input_seq_len)

        return extra_ret

    @classmethod
    def ignore_save_variables(cls):
        #return []
        return ["prev_targets", "prev_target_seq_len", "prev_dlg_ids"]
    
    def output_result(self, ground_truth_labels, predicted_labels,
            ground_truth_label_len, predicted_label_len, extra_ops, eval_count):
        for gt_da, pr_da, extra in zip(ground_truth_labels[0], predicted_labels[0],
                extra_ops[1]):
            with open(self.hparams.result_output_file, "a") as f:
                f.write('\t'.join([
                    str(gt_da),
                    str(pr_da),
                    ' '.join([str(f) for f in extra])
                ]) + '\n')
