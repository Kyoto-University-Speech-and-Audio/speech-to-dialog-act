import random

import tensorflow as tf

from .base import BaseInputData

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        self.mode = mode
        self.hparams = hparams

        BaseInputData.__init__(self, hparams, mode)

        inputs = []
        for line in open(self.data_filename, "r"):
            if line.strip() == "": continue
            [filename, is_first_utt, speaker, target] = line.strip().split('\t')
            inputs.append((filename, is_first_utt == '1', int(speaker), target))

        self.size = len(inputs)
        self.inputs = inputs

    def init_dataset(self):
        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)
        self.is_first_utts = tf.placeholder(dtype=tf.bool, shape=[None])
        self.speakers = tf.placeholder(dtype=tf.int32, shape=[None])

        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        src_is_first_utts_dataset = tf.data.Dataset.from_tensor_slices(self.is_first_utts)
        src_speakers_dataset = tf.data.Dataset.from_tensor_slices(self.speakers)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, src_is_first_utts_dataset, src_speakers_dataset))
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, src_is_first_utts_dataset, src_speakers_dataset, tgt_dataset))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset.take(10)

        self.batched_dataset = src_tgt_dataset.padded_batch(
            self.batch_size,
            padded_shapes=(([], [None, self.hparams.num_features], []), [], [],
                           ([None], [])),
            padding_values=(('', 0.0, 0), False, -1, (1, 0))
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]

        filenames, is_first_utts, speakers, targets = zip(*inputs)
        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets,
            self.is_first_utts: is_first_utts,
            self.speakers: speakers
        })