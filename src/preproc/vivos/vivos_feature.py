import os
from tensorflow.python.platform import gfile
import numpy as np

from csp.utils import wav_utils

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import numpy as np
from struct import unpack, pack
from ..utils import utils
import random
from .base import BaseInputData

DCT_COEFFICIENT_COUNT = 40

class BatchedInput(BaseInputData):
    num_features = DCT_COEFFICIENT_COUNT
    
    def __init__(self, hparams, mode):
        self.mode = mode
        self.hparams = hparams

        if hparams.input_unit == 'char':
            chars = [s.strip().split(' ', 1) for s in open('data/vivos/chars.txt', encoding='utf-8')]
            self.decoder_map = {int(char[0]): char[1] for char in chars}
            self.num_classes = len(chars)
        else:
            words = [s.strip().split(' ', 1) for s in open('data/vivos/words.txt', encoding='utf-8')] 
            self.decoder_map = {int(word[0]): word[1] for word in words}
            self.num_classes = len(words)

        BaseInputData.__init__(self, hparams, mode)

        filenames, targets = [], []
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            data_filename = "data/vivos/train/data_chars.txt" if self.hparams.input_unit == 'char' \
                else "data/vivos/train/data.txt"
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            data_filename = "data/vivos/test/data_chars.txt" if self.hparams.input_unit == "char" \
                else "data/vivos/test/data.txt"
        else:
            data_filename = "data/vivos/infer/test.txt"

        for line in open(data_filename):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                filename, target = line.strip().split(' ', 1)
                targets.append(target)
            else:
                filename = line.strip()
            filenames.append(filename)
        self.size = len(filenames)
        self.input_filenames = filenames
        self.input_targets = targets

    def init_dataset(self):
        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)

        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename:
                tf.cast(tf.py_func(self.load_input, [filename], tf.float64),
                    tf.float32))
        src_dataset = src_dataset.map(lambda feat: (feat, tf.shape(feat)[0]))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = src_dataset
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        if self.mode == tf.estimator.ModeKeys.TRAIN and self.hparams.max_train > 0:
            src_tgt_dataset.take(self.hparams.max_train)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset.take(10)

        self.batched_dataset = utils.get_batched_dataset(
            src_tgt_dataset,
            self.hparams.batch_size,
            DCT_COEFFICIENT_COUNT,
            self.hparams.num_buckets, self.mode,
            padding_values=0 if self.hparams.input_unit == "char" else 1
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def init_from_wav_files(self, wav_filenames):
        src_dataset = tf.data.Dataset.from_tensor_slices(wav_filenames)
        src_dataset = wav_utils.wav_to_features(src_dataset, self.hparams, 40)
        src_dataset = src_dataset.map(lambda feat: (feat, tf.shape(feat)[0]))

        self.batched_dataset = utils.get_batched_dataset(
            src_dataset,
            self.hparams.batch_size,
            DCT_COEFFICIENT_COUNT,
            self.hparams.num_buckets, self.mode
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def load_input(self, filename):
        return np.load(filename.decode('utf-8') + '.npy').astype(float)

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def reset_iterator(self, sess, skip=0, shuffle=False):
        filenames = self.input_filenames
        targets = self.input_targets
        
        if shuffle:
            bucket = 100
            shuffled_filenames = []
            shuffled_targets = []
            start, end = 0, 0
            for i in range(0, len(filenames) // bucket):
                start, end = i * bucket, min((i + 1) * bucket, len(filenames))
                ls = list(zip(filenames[start:end], targets[start:end]))
                random.shuffle(ls)
                fs, ts = zip(*ls)
                shuffled_filenames += fs
                shuffled_targets += ts
            filenames = shuffled_filenames
            targets = shuffled_targets

        filenames = filenames[skip:]
        targets = targets[skip:]

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets
        })

    def decode(self, d):
        return d
        ret = ''
        for c in d:
            if c <= 0: continue
            if self.hparams.input_unit == "word":
                if c == 1: return ret # sos
            # ret += str(c) + " "
            blank = ' ' if self.hparams.input_unit == "word" else ''
            ret += self.decode_map[c] + " " + blank if c in self.decoder_map else '?'
        return ret
