import os
import random
from struct import unpack
from subprocess import call, PIPE

import numpy as np
import tensorflow as tf
from ..utils import utils
import ffmpeg


class BaseInputData():
    def __init__(self,
                 hparams,
                 mode,
                 batch_size,
                 dev=False):
        self.hparams = hparams
        self.mode = mode
        self.vocab = self.load_vocab(hparams.vocab_file)
        self.vocab_size = len(self.vocab)

        if hparams.norm_mean_path is not None:
            mean = open(hparams.norm_mean_path).read().split('\n')[:-1]
            self.mean = np.array([float(x) for x in mean])
            var = open(hparams.norm_var_path).read().split('\n')[:-1]
            self.var = np.array([float(x) for x in var])
        else:
            self.mean = self.var = None

        hparams.vocab_size = self.vocab_size

        self.batch_size = tf.cast(batch_size, tf.int64)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.data_filename = hparams.train_data
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            self.data_filename = hparams.test_data if not dev else hparams.dev_data
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            self.data_filename = hparams.input_path

        self.filenames = tf.placeholder(dtype=tf.string)
        self.targets = tf.placeholder(dtype=tf.string)

        inputs = []
        with open(self.data_filename, "r") as f:
            headers = f.readline().split('\t')
            for line in f.read().split('\n')[1:]:
                if self.mode != tf.estimator.ModeKeys.PREDICT:
                    if line.strip() == "": continue
                    input = {headers[i]: dat for i, dat in enumerate(line.strip().split('\t'))}
                    
                    if hparams.get('use_sos_eos', False) and 'target' in input: 
                        input['target'] = "%d %s %d" % (
                            self.hparams.sos_index, input['target'], 
                            self.hparams.eos_index)
                    inputs.append(input)

        self.size = len(inputs)
        self.inputs = inputs

    def load_vocab(self, vocab_file):
        labels = []
        count = 0
        for s in open(vocab_file, encoding=self.hparams.encoding).read().split('\n'):
            if ' ' in s:  # word id
                labels.append(s.strip().split(' ', 1))
            else:  # word
                labels.append((s, str(count)))
                count += 1
        
        vocab = {int(id): label for label, id in labels}
        if self.hparams.get('use_sos_eos', False):
            self.hparams.eos_index = len(labels)
            self.hparams.sos_index = len(labels) + 1
            vocab[len(labels)] = '<eos>'
            vocab[len(labels) + 1] = '<sos>'
        elif self.hparams.get('eos_index', None) is None:
            self.hparams.eos_index = len(labels)
            vocab[len(labels)] = '<eos>'

        return vocab

    def get_batched_dataset(self, dataset):
        return utils.get_batched_dataset(
            dataset,
            self.batch_size,
            self.hparams.num_features,
            # self.hparams.num_buckets,
            self.mode,
            padding_values=self.hparams.get("eos_index", len(self.vocab) - 1)
        )

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

        if self.mean is None:
            dat = (dat - dat.mean()) / np.std(dat)
        else:
            dat = (dat - self.mean) / np.sqrt(self.var)

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
        # return np.array([[0.0]], dtype=np.float32)
        dat = np.load(filename.decode('utf-8')).astype(np.float32)
        return dat

    def load_input(self, filepath):
        ext = os.path.splitext(filepath)[1]
        if ext == b".htk":
            return self.load_htk(filepath)
        elif ext == b".wav":
            return self.load_wav(filepath)
        elif ext == b".npy":
            return self.load_npy(filepath)
        elif ext in {b'.webm'}:
            stream = ffmpeg.input(filepath)
            output_path = os.path.join('tmp', 'output.wav')
            stream = ffmpeg.output(stream, output_path)
            ffmpeg.run(stream)
            dat = self.load_wav(output_path)
            os.remove(output_path)
            return dat
        else:
            return np.array([[0.0] * 120] * 8).astype(np.float32)

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]

    def get_word(self, id):
        return self.vocab[id]

    def decode(self, d, id):
        """Decode from label ids to words"""
        ret = []
        for c in d:
            if c < 0: continue
            if self.vocab[c] == '<eos>': return ret  # sos
            if self.vocab[c] == '<sos>': continue
            val = self.get_word(c)
            if val != '':
                ret.append(val if c in self.vocab else '?')
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

    def get_inputs_list(self, inputs, field):
        return [inp[field] for inp in self.inputs]
