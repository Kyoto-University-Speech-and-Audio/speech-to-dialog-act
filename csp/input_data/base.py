import tensorflow as tf
import numpy as np
from subprocess import call, PIPE
import os
from struct import unpack

class BaseInputData():
    def __init__(self, hparams, mode):
        labels = [s.strip().split(' ', 1) for s in open(hparams.vocab_file, encoding='eucjp')]
        self.decoder_map = {int(label[1]): label[0] for label in labels}
        self.num_classes = len(labels)
        hparams.num_classes = self.num_classes

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.data_filename = hparams.train_data
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.data_filename = hparams.eval_data
        else:
            self.data_filename = hparams.input_path

    def load_wav(self, filename):
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

    def load_input(self, filename):
        if os.path.splitext(filename)[1] == b".htk":
            return self.load_htk(filename)
        else:
            return self.load_wav(filename)