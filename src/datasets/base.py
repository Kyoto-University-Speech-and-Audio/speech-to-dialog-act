import os
import random
from struct import unpack
from subprocess import call, PIPE

import numpy as np
import tensorflow as tf

class BaseInputData():
    def __init__(self,
                 hparams,
                 mode,
                 mean_val_path=None,
                 var_val_path=None):

        labels = [s.strip().split(' ', 1) for s in open(hparams.vocab_file, encoding=hparams.encoding)]
        self.decoder_map = {int(label[1]): label[0] for label in labels}
        self.num_classes = len(labels)
        self.hparams = hparams
        self.mode = mode

        if mean_val_path is not None:
            mean = open(mean_val_path).read().split('\n')[:-1]
            self.mean = np.array([float(x) for x in mean])
            var = open(var_val_path).read().split('\n')[:-1]
            self.var = np.array([float(x) for x in var])

        hparams.num_classes = self.num_classes

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.data_filename = hparams.train_data
            self.batch_size = hparams.batch_size
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.data_filename = hparams.eval_data
            self.batch_size = hparams.eval_batch_size
        else:
            self.data_filename = hparams.input_path

    def load_wav(self, filename):
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

        dat = (dat - self.mean) / np.sqrt(self.var)
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
        if os.path.splitext(filename)[1] == b".htk":
            return self.load_htk(filename)
        elif os.path.splitext(filename)[1] == b".wav":
            return self.load_wav(filename)
        elif os.path.splitext(filename)[1] == b".npy":
            return self.load_npy(filename)

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def decode(self, d):
        """Decode from label ids to words"""
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
