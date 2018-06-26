import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils


DCT_COEFFICIENT_COUNT = 120

class BatchedInput(BaseInputData):
    num_features = DCT_COEFFICIENT_COUNT
    # num_classes = 3260 + 1
    # num_classes = 34331
    
    def __init__(self, hparams, mode):
        self.mode = mode
        self.hparams = hparams

        BaseInputData.__init__(self, hparams, mode)

        filenames, targets, contexts = [], [], []
        for line in open(self.data_filename, "r"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                [filename, context, target] = line.strip().split('\t')
                targets.append(target)
                contexts.append([float(c) for c in context.split(' ')])
            else:
                filename = line.strip()
            filenames.append(filename)
        self.size = len(filenames)
        self.input_filenames = filenames
        self.input_targets = targets
        self.input_contexts = contexts

    def init_dataset(self):
        CONTEXT_SIZE = 35
        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)
        self.contexts = tf.placeholder(dtype=tf.float32, shape=[None, CONTEXT_SIZE])

        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        src_context_dataset = tf.data.Dataset.from_tensor_slices(self.contexts)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, src_context_dataset))
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, src_context_dataset, tgt_dataset))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset.take(10)

        self.batched_dataset = src_tgt_dataset.padded_batch(
            self.hparams.batch_size,
            padded_shapes=(([], [None, DCT_COEFFICIENT_COUNT], []), [CONTEXT_SIZE],
                           ([None], [])),
            padding_values=(('', 0.0, 0), 0.0, (1, 0))
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def reset_iterator(self, sess, skip=0, shuffle=False):
        filenames = self.input_filenames
        targets = self.input_targets
        contexts = self.input_contexts
        
        if shuffle and 0:
            bucket = 10000
            shuffled_filenames = []
            shuffled_targets = []
            for i in range(0, len(filenames) // bucket):
                start, end = i * bucket, min((i + 1) * bucket, len(filenames))
                ls = list(zip(filenames[start:end], targets[start:end]))
                random.shuffle(ls)
                fs, ts = zip(*ls)
                shuffled_filenames += fs
                shuffled_targets += ts
            filenames = shuffled_filenames
            targets = shuffled_targets

        if shuffle:
            ls = list(zip(filenames, targets))
            random.shuffle(ls)
            filenames, targets = zip(*ls)

        filenames = filenames[skip:]
        targets = targets[skip:]
        contexts = contexts[skip:]

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets,
            self.contexts: contexts
        })