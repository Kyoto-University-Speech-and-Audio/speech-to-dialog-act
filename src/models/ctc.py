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
    def __init__(self):
        super().__init__()

    def __call__(self, hparams, mode, batched_input, **kwargs):
        BaseModel.__call__(self, hparams, mode, batched_input, **kwargs)
        return self

    def get_ground_truth_label_placeholder(self): return [self.targets]
    def get_predicted_label_placeholder(self): return [self.decoded[0]]

    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len), (self.targets, self.target_seq_len)) = \
                self.iterator.get_next()
        else:
            self.input_filenames, self.inputs, self.input_seq_len = self.iterator.get_next()

    def _build_graph(self):
        # generate a SparseTensor required by ctc_loss op.
        self.targets = ops_utils.sparse_tensor(self.targets, padding_value=-1)

        cells_fw = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
        cells_bw = [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,
                                                                       cells_bw, self.inputs,
                                                                       sequence_length=self.input_seq_len,
                                                                       dtype=tf.float32)

        # Reshaping to apply the same weights over the timesteps
        # outputs = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.layers.dense(outputs, self.hparams.vocab_size)
        # Reshaping back to the original shape
        # logits = tf.reshape(logits, [hparams.batch_size, -1, self.vocab_size])

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

    def get_extra_ops(self):
        return []
