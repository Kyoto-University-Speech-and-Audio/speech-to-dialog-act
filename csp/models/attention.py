#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os

import tensorflow as tf
import numpy as np

import attention_configs as configs
import attention_input_data as input_data
from attention_input_data import BatchedInput

from tensorflow.python.layers import core as layers_core

from six.moves import xrange as range

num_units = 50 # Number of units in the LSTM cell

# Hyper-parameters
num_epochs = 10000
num_hidden = 50
num_encoder_layers = 1
num_decoder_layers = 1
batch_size = configs.BATCH_SIZE
initial_learning_rate = 1e-2
momentum = 0.9

max_gradient_norm = 5.0

num_batches_per_epoch = configs.TRAINING_SIZE // batch_size

class AttentionModel():
    def __init__(self, hparams, mode, iterator):
        self.iterator = iterator
        self.mode = mode

        ((self.inputs, self.input_seq_len), (self.target_labels, self.target_seq_len)) = \
            iterator.get_next()

        self.targets = tf.one_hot(self.target_labels, depth=input_data.NUM_LABELS)

        # Projection
        self.output_layer = layers_core.Dense(input_data.NUM_LABELS, use_bias=False, name="output_projection")

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

        loss = tf.reduce_sum(crossent) / tf.to_float(batch_size)

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

            self.embedding_decoder = tf.diag(tf.ones(input_data.NUM_LABELS))
            if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.target_seq_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)

            elif self.mode == tf.estimator.ModeKeys.PREDICT:
                start_tokens = tf.fill([configs.BATCH_SIZE], input_data.TGT_SOS_INDEX)
                end_token = input_data.TGT_EOS_INDEX

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
        _, loss = sess.run([self.update_step, self.loss])
        return loss


    def predict(self, sess):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        _, sample_id = sess.run([self.logits, self.sample_id])
        return sample_id


def train(hparams):
    tf.reset_default_graph()

    batched_input = BatchedInput()
    infer_input = BatchedInput()

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('test')

    # val_inputs, val_seq_len, val_targets = audio_processor.next_train_batch(0, 1)

    # train_sess = tf.Session(graph=train_model.graph)
    # predict_sess = tf.Session(graph=predict_model.graph)

    with tf.Session() as sess:
        batched_input.get_data(sess)
        infer_input.get_data(sess)
        iterator = batched_input.batched_dataset.make_initializable_iterator()
        infer_iterator = infer_input.batched_dataset.make_initializable_iterator()
        with tf.variable_scope('root'):
            train_model = Model(hparams, tf.estimator.ModeKeys.TRAIN, iterator=iterator)

        with tf.variable_scope('root', reuse=True):
            predict_model = Model(hparams, tf.estimator.ModeKeys.PREDICT, iterator=infer_iterator)

        sess.run(iterator.initializer)
        sess.run(infer_iterator.initializer)
        sess.run(tf.global_variables_initializer())

        global_step = 0
        while global_step < num_epochs:
            start_time = time.time()
            try:
                sess.run(infer_iterator.initializer)
                loss = train_model.train(sess)
                print("Loss: %.3f" % (loss))
                sample_words = predict_model.predict(sess)
                print([batched_input.decode(words) for words in sample_words])
            except tf.errors.OutOfRangeError:
                global_step += 1
                sess.run(iterator.initializer)