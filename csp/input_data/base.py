import tensorflow as tf
import numpy as np
from subprocess import call, PIPE
import os
from struct import unpack
from ..utils import wav_utils, utils
import random

DCT_COEFFICIENT_COUNT = 120

class BaseInputData():
    def __init__(self, hparams, mode):
        labels = [s.strip().split(' ', 1) for s in open(hparams.vocab_file, encoding='eucjp')]
        self.decoder_map = {int(label[1]): label[0] for label in labels}
        self.num_classes = len(labels)
        self.hparams = hparams
        self.mode = mode

        hparams.num_classes = self.num_classes

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.data_filename = hparams.train_data
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.data_filename = hparams.eval_data
        else:
            self.data_filename = hparams.input_path

    @property
    def num_features(self):
        return DCT_COEFFICIENT_COUNT

    def load_wav(self, filename):
        print(filename)
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
        # print("spam", spam)
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

    def load_htk(self, filename):
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

    def load_npy(self, filename):
        return np.load(filename.decode('utf-8')).astype(np.float32)

    def load_input(self, filename):
        # print(filename)
        if os.path.splitext(filename)[1] == b".htk":
            return self.load_htk(filename)
        elif os.path.splitext(filename)[1] == b".wav":
            return self.load_wav(filename)
        elif os.path.splitext(filename)[1] == b".npy":
            return self.load_npy(filename)

    def init_from_wav_files(self, wav_filenames):
        src_dataset = tf.data.Dataset.from_tensor_slices(wav_filenames)
        src_dataset = wav_utils.wav_to_features(src_dataset, self.hparams, 40)
        src_dataset = src_dataset.map(lambda feat: (feat, tf.shape(feat)[0]))

        self.batched_dataset = utils.get_batched_dataset(
            src_dataset,
            self.hparams.batch_size,
            self.num_features,
            self.hparams.num_buckets, self.mode
        )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def decode(self, d):
        ret = []
        for c in d:
            if c <= 0: continue
            if self.hparams.input_unit == "word":
                if c == 1: return ret # sos
            # ret += str(c) + " "
            if self.decoder_map[c] == '<sos>': continue
            if self.hparams.input_unit == "word":
                val = self.decoder_map[c].split('+')[0]
            else:
                val = self.decoder_map[c]
            ret.append(val if c in self.decoder_map else '?')
        return ret

    def shuffle(self, inputs, bucket_size=None):
        if bucket_size:
            shuffled_inputs = []
            for i in range(0, len(inputs) // bucket_size):
                start, end = i * bucket_size, min((i + 1) * bucket_size, len(inputs))
                ls = inputs[start:end]
                random.shuffle(ls)
                shuffled_inputs += ls
            return shuffled_inputs
        else:
            ls = list(inputs)
            random.shuffle(ls)
            return ls