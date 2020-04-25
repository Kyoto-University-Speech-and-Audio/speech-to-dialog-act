import random

import tensorflow as tf
import numpy as np

from .base import BaseInputData
from ..utils import utils
import os, glob

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        BaseInputData.__init__(self, hparams, mode)

        #for line in open(self.data_filename, "r"):
        #    if self.mode != tf.estimator.ModeKeys.PREDICT:
        #        if line.strip() == "": continue
        #        filename, target = line.strip().split(' ', 1)
        #        inputs.append((filename, "%d %s %d" % (self.hparams.sos_index, target, self.hparams.eos_index)))
        #    else:
        #        filename = line.strip()
        #        inputs.append(filename)

        #self.size = len(inputs)
        self.filenames = glob.glob(os.path.join(self.data_filename, '*.tfrecords'))
        self.size = None

    def load_vocab(self, vocab_file):
        labels = [s.strip() for s in open(vocab_file, encoding=self.hparams.encoding)]
        self.vocab = {id: label for id, label in enumerate(labels)}
        self.vocab_size = len(labels) + 2
        self.hparams.eos_index = self.vocab_size - 2
        self.hparams.sos_index = self.vocab_size - 1
        self.vocab[self.vocab_size - 2] = '<eos>'
        self.vocab[self.vocab_size - 1] = '<sos>'

    def _parse_features(self, example):
        features = {
            "input": tf.VarLenFeature(tf.float32),
            "target": tf.VarLenFeature(tf.int64)
        }
        parsed_features = tf.parse_single_example(example, features)
        input = tf.reshape(tf.sparse_tensor_to_dense(parsed_features['input']), [-1, self.hparams.num_features])
        target = tf.concat([
            tf.constant([self.hparams.sos_index]),
            tf.cast(tf.sparse_tensor_to_dense(parsed_features['target']), tf.int32),
            tf.constant([self.hparams.eos_index])], 0)

        return input, target

    def init_dataset(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_features)
        dataset = dataset.map(lambda input, target: ((tf.constant('abc'), input, tf.shape(input)[0]), (target, tf.shape(target)[0])))

        #src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        #src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        #src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        #if self.mode == tf.estimator.ModeKeys.PREDICT:
        #    src_tgt_dataset = src_dataset
        #else:
        #    tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
        #    tgt_dataset = tgt_dataset.map(
        #        lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
        #    tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

        #    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        self.dataset = dataset
        batched_dataset = self.get_batched_dataset(dataset)
        self.iterator = tf.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
        # self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        sess.run(self.iterator.make_initializer(self.get_batched_dataset(self.dataset)))
