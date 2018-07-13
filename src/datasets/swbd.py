import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        BaseInputData.__init__(self, hparams, mode)

        inputs = []
        for line in open(self.data_filename, "r"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                filename, target = line.strip().split(' ', 1)
                inputs.append((filename, "%d %s %d" % (self.hparams.sos_index, target, self.hparams.eos_index)))
            else:
                filename = line.strip()
                inputs.append(filename)

        self.size = len(inputs)
        self.inputs = inputs

    def load_vocab(self, vocab_file):
        labels = [s.strip() for s in open(vocab_file, encoding=self.hparams.encoding)]
        self.decoder_map = {id: label for id, label in enumerate(labels)}
        self.num_classes = len(labels) + 2
        self.hparams.eos_index = self.num_classes - 2
        self.hparams.sos_index = self.num_classes - 1
        self.decoder_map[self.num_classes - 2] = '<eos>'
        self.decoder_map[self.num_classes - 1] = '<sos>'

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

        self.batched_dataset = self.get_batched_dataset(src_tgt_dataset)
        self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, str):
        return [[self.hparams.sos_index] + [int(x) for x in str.decode('utf-8').split(' ')] + [self.hparams.eos_index]]

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]
        filenames, targets = zip(*inputs)
        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets
        })
