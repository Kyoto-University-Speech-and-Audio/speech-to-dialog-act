#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os

from .base import BaseModel

import tensorflow as tf
import pdb
from six.moves import xrange as range

from ..utils import ops_utils

num_units = 320
num_layers = 3

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('test')

class CTCModel(BaseModel):
    num_classes = 3260 + 1
    def __init__(self, hparams, mode, iterator):
        BaseModel.__init__(self, hparams, mode, iterator)
        self.num_classes = self.hparams.num_classes
        if self.train_mode:
            self.train_summary = tf.summary.merge([self.train_summary, tf.summary.scalar("train_label_error_rate", self.ler)])

    def _build_graph(self):
        # generate a SparseTensor required by ctc_loss op.
        self.targets = ops_utils.sparse_tensor(self.target_labels, padding_value=-1)

        # Defining the cell
        # cells = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
        # stack = tf.contrib.rnn.MultiRNNCell(cells)

        # The second output is the last state and we will no use that
        # outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.input_seq_len, dtype=tf.float32)

        cells_fw = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
        cells_bw = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                       cells_bw, self.inputs,
                                                                       sequence_length=self.input_seq_len,
                                                                       dtype=tf.float32)

        # Reshaping to apply the same weights over the timesteps
        # outputs = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.layers.dense(outputs, self.num_classes)
        # Reshaping back to the original shape
        # logits = tf.reshape(logits, [hparams.batch_size, -1, self.num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        self.logits = logits

        loss = tf.nn.ctc_loss(self.targets, logits, self.input_seq_len, ignore_longer_outputs_than_inputs=True)
        self.loss = tf.reduce_mean(loss)

        self.decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.input_seq_len)
        # self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, self.input_seq_len)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

        # label error rate
        self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32),
                                                   self.targets))

        return self.loss

    def train(self, sess):
        # inputs, targets = sess.run([self.inputs, self.targets])
        batch_lost, _, self.summary, _ler, dense_decoded, \
            inputs, labels, inputs_len, labels_len, logits, global_step = \
            sess.run([
                self.loss,
                self.update,
                self.train_summary,
                self.ler,
                self.dense_decoded,
                self.inputs, self.target_labels, self.input_seq_len,
                self.target_seq_len, self.logits,
                self.global_step
            ])

        return batch_lost, global_step

    def eval(self, sess):
        target_labels, loss, ler, decoded, self.summary = \
            sess.run([
                self.target_labels,
                self.loss,
                self.ler,
                self.dense_decoded,
                self.summary
            ])
        return target_labels, loss, decoded

