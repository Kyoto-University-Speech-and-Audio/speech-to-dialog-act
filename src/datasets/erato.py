import os
from tensorflow.python.platform import gfile
import numpy as np

from ..utils import wav_utils

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import numpy as np
from struct import unpack, pack
from ..utils import utils
import random
from .base import BaseInputData

from subprocess import call, PIPE

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = 1

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        self.mode = mode
        self.hparams = hparams

        if hparams.input_unit == 'char':
            chars = [s.strip().split(' ', 1) for s in open('data/aps-sps/char.id', encoding='eucjp')]
            self.decoder_map = {int(char[1]): char[0] for char in chars}
            self.num_classes = len(chars) + 1
        else:
            words = [s.strip().split(' ', 1) for s in open('data/aps-sps/word.id', encoding='eucjp')]
            self.decoder_map = {int(word[1]): word[0].split('+')[0] for word in words}
            self.num_classes = len(words)

        BaseInputData.__init__(self, hparams, mode)

        filenames, targets = [], []
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            data_filename = "data/erato/train.txt" if self.hparams.input_unit == 'char' \
                else "data/erica/train.txt"
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            data_filename = "data/erato/eval.txt" if self.hparams.input_unit == "char" \
                else "data/erica/eval.txt"
        else:
            data_filename = hparams.input_path

        for line in open(data_filename, "r"):
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
        if os.path.splitext(filename)[1] == b".htk":
            fh = open(filename, "rb")
            spam = fh.read(12)
            nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
            veclen = int(sampSize / 4)
            fh.seek(12, 0)
            dat = np.fromfile(fh, dtype=np.float32)
            if len(dat) % veclen != 0: dat = dat[:len(dat) - len(dat) % veclen]
            dat = dat.reshape(len(dat) // veclen, veclen)
            dat = dat.byteswap()
            fh.close()
            return dat
        else:
            mean = open("data/aps-sps/mean.dat").read().split('\n')[:-1]
            mean = np.array([float(x) for x in mean])
            var = open("data/aps-sps/var.dat").read().split('\n')[:-1]
            var = np.array([float(x) for x in var])

            outfile = "tmp.htk"
            call([
                self.hparams.hcopy_path,
                "-C", self.hparams.hcopy_config, "-T", "1",
                filename, outfile
            ], stdout=PIPE)
            fh = open(outfile, "rb")
            spam = fh.read(12)
            nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
            veclen = int(sampSize / 4)
            fh.seek(12, 0)
            dat = np.fromfile(fh, dtype=np.float32)
            dat = dat.reshape(len(dat) // veclen, veclen)
            dat = dat.byteswap()
            fh.close()

            dat = (dat - mean) / np.sqrt(var)
            fh.close()

            return np.float32(dat)


    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def reset_iterator(self, sess, skip=0, shuffle=False):
        filenames = self.input_filenames
        targets = self.input_targets
        
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

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: filenames,
            self.targets: targets
        })

    def decode(self, d):
        ret = []
        for c in d:
            if c <= 0: continue
            if self.hparams.input_unit == "word":
                if c == 1: return ret # sos
            # ret += str(c) + " "
            if self.decoder_map[c] == '<sos>': continue
            ret.append(self.decoder_map[c] if c in self.decoder_map else '?')
        return ret
