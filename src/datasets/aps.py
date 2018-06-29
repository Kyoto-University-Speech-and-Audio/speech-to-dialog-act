import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        BaseInputData.__init__(
            self, hparams, mode,
            mean_val_path="data/aps-sps/mean.dat",
            var_val_path="data/aps-sps/var.dat")

        inputs = []
        for line in open(self.data_filename, "r"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                filename, target = line.strip().split(' ', 1)
                inputs.append((filename, target))
            else:
                filename = line.strip()
                inputs.append(filename)

        self.size = len(inputs)
        self.inputs = inputs

        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)

    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = src_dataset
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset.take(10)

        self.batched_dataset = utils.get_batched_dataset_bucket(
            src_tgt_dataset,
            self.hparams.batch_size,
            self.hparams.num_features,
            self.hparams.num_buckets,
            self.mode,
            padding_values=0 if self.hparams.input_unit == "char" else 1
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]

        filenames, targets = zip(*inputs)

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets
        })
