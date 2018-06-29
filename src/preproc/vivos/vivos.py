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

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = 1

SRC_EOS_ID = 0

DCT_COEFFICIENT_COUNT = 120

class BatchedInput():
    num_features = DCT_COEFFICIENT_COUNT
    num_classes = 3260 + 1
    
    def __init__(self, hparams, mode):
        self.mode = mode
        self.hparams = hparams
        self.size = 50
        chars = [s.strip().split(' ', 1) for s in
                    open('/n/rd32/mimura/e2e/data/script/aps/char.id',
                        encoding='eucjp')]
        self.decoder_map = {int(char[1]): char[0] for char in chars}

    def init_dataset(self):
        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)

        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: tf.py_func(self.load_input, [filename], tf.float32))
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

        src_tgt_dataset.shuffle(1000)
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset.take(10)

        self.batched_dataset = utils.get_batched_dataset(
            src_tgt_dataset,
            self.hparams.batch_size,
            DCT_COEFFICIENT_COUNT,
            self.hparams.num_buckets, self.mode)

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def load_input(self, filename):
        fh = open(filename, "rb")
        spam = fh.read(12)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype=np.float32)
        dat = dat.reshape(int(len(dat) / veclen), veclen)
        dat = dat.byteswap()
        fh.close()
        return dat

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def reset_iterator(self, sess):
        filenames, targets = [], []
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            data_filename = "data/aps-sps/train/char_sort_xlen.txt"
        else:
            data_filename = "data/aps-sps/test/targets.txt"
        for line in open(data_filename):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                filename, target = line.strip().split(' ', 1)
                targets.append(target)
            else:
                filename = line.strip()
            filenames.append(filename)
        self.size = len(filenames)
        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets
        })

    def decode(self, d):
        ret = ''
        for c in d:
            if c > 0:
                ret += self.decoder_map[c] + '' if c in self.decoder_map else '?'
        return ret
