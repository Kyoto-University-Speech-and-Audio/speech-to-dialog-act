import os
from tensorflow.python.platform import gfile
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

DATA_DIR = os.path.join("..", "data", "vivos")

CHARS = list("aăâeêioôơuưy") + \
        list("áắấéếíóốớúứý") + \
        list("àằầèềìòồờùừỳ") + \
        list("ảẳẩẻểỉỏổởủửỷ") + \
        list("ãẵẫẽễĩõỗỡũữỹ") + \
        list("ạặậẹệịọộợụựỵ") + \
        list("bcdđghiklmnpqrstvxwz") + \
        ["ch", "tr", "th", "ng", "nh"]
CHARS = [(-len(s), s) for s in CHARS]
CHARS.sort()
CHARS = [s for _, s in CHARS]

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = 1

TGT_SOS_TOKEN = '<s>'
TGT_EOS_TOKEN = '</s>'
TGT_SOS_INDEX = 1
TGT_EOS_INDEX = 2

SRC_EOS_ID = 0

DCT_COEFFICIENT_COUNT = 40

TRAINING_SIZE = 11520
#TRAINING_SIZE = 1
TRAINING_START = 0

WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 10.0

SAMPLE_RATE = 16000

class BatchedInput():
    def __init__(self, mode, batch_size):
        # self.prepare_inputs()
        self.batch_size = batch_size

        mode_dir = "train" if mode == tf.estimator.ModeKeys.TRAIN else "test"

        search_path = os.path.join(DATA_DIR, mode_dir, "waves", '**', '*.wav')
        self.ls_input_files = gfile.Glob(search_path)
        self.ls_input_files.sort()

        tgt_dataset = tf.data.TextLineDataset(os.path.join(DATA_DIR, mode_dir, "prompts_pp.txt")).skip(TRAINING_START)
        # tgt_dataset = tgt_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).take(1))
        # tgt_dataset = tgt_dataset.map(lambda string: tf.string_split([string], delimiter="").values)
        tgt_dataset = tgt_dataset.map(lambda str: tf.py_func(self.encode, [str], tf.int64))
        tgt_dataset = tgt_dataset.map(lambda phones: (tf.cast(phones, tf.int32), tf.size(phones)))

        # src_dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32))
        self.input_files = tf.placeholder(tf.string, shape=[None])
        src_dataset = tf.data.Dataset.from_tensor_slices(self.input_files).skip(TRAINING_START)
        src_dataset = src_dataset.map(lambda filename: io_ops.read_file(filename))
        src_dataset = src_dataset.map(lambda wav_loader: contrib_audio.decode_wav(wav_loader, desired_channels=1))
        src_dataset = src_dataset.map(lambda wav_decoder:
                                      (contrib_audio.audio_spectrogram(
                                          wav_decoder.audio,
                                          window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
                                          stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
                                          magnitude_squared=True), wav_decoder.sample_rate))
        src_dataset = src_dataset.map(lambda spectrogram, sample_rate: contrib_audio.mfcc(
            spectrogram, sample_rate,
            dct_coefficient_count=DCT_COEFFICIENT_COUNT))
        src_dataset = src_dataset.map(lambda inputs: (
            inputs,
            tf.nn.moments(inputs, axes=[1])
        ))
        src_dataset = src_dataset.map(lambda inputs, moments: (
            tf.divide(tf.subtract(inputs, moments[0]), moments[1]),
            tf.shape(inputs)[1]
        ))
        src_dataset = src_dataset.map(lambda inputs, seq_len: (
            inputs[0],
            seq_len
        ))

        if TRAINING_SIZE > 0:
            src_dataset = src_dataset.take(TRAINING_SIZE)
            tgt_dataset = tgt_dataset.take(TRAINING_SIZE)

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        self.batched_dataset = src_tgt_dataset
        self.batched_dataset = src_tgt_dataset.padded_batch(
            self.batch_size,
            padded_shapes=(([None, DCT_COEFFICIENT_COUNT], []),
                           ([None], [])))

        self.iterator = self.batched_dataset.make_initializable_iterator()

    num_features = DCT_COEFFICIENT_COUNT
    num_classes = len(CHARS) + FIRST_INDEX + 1

    def __len__(self):
        return TRAINING_SIZE

    def prepare_inputs(self):
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)

        spectrogram = contrib_audio.audio_spectrogram(
            wav_decoder.audio,
            window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
            stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
            magnitude_squared=True)
        inputs = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=DCT_COEFFICIENT_COUNT)
        mean, var = tf.nn.moments(inputs, axes=[1])
        self.inputs_ = tf.divide(tf.subtract(inputs, mean), var)
        self.seq_len_ = tf.shape(self.inputs_)[1]

    @classmethod
    def encode(cls, s):
        s = s.decode('utf-8')
        s = ' '.join(str(s).strip().lower().split(' ')).replace('.', '').replace(',', '')
        # s = ' '.join(s.split(' ')[1:])
        s = s.replace(' ', '  ')
        targets = s.split(' ')

        target_labels = []
        for x in targets:
            if x == '': target_labels.append(SPACE_INDEX)
            else:
                start = 0
                end = len(x) + 1
                while end >= start and start < len(x):
                    end -= 1
                    if x[start:end] in CHARS:
                        target_labels.append(FIRST_INDEX + CHARS.index(x[start:end]))
                        start = end
                        end = len(x) + 1

        return [target_labels]

    def reset_iterator(self, sess):
        sess.run(self.iterator.initializer, feed_dict={ self.input_files: self.ls_input_files })

    @classmethod
    def decode(cls, d):
        str_decoded = ''.join([CHARS[x - FIRST_INDEX] if x >= FIRST_INDEX else ' ' for x in d])
        return str_decoded


# Preprocessing
'''
if __name__ == "__main__":
    search_path = os.path.join("..", DATA_DIR, "test", "waves", '**', '*.wav')
    ls_input_files = gfile.Glob(search_path)
    ls_input_files.sort()

    d = {}
    with open(os.path.join("..", DATA_DIR, "test", "prompts.txt"), encoding="utf-8") as f:
        d = { s.split(' ')[0]: ' '.join(s.split(' ')[1:]) for s in f.read().split('\n') }

    sents = [d[os.path.splitext(os.path.basename(path))[0]] for path in ls_input_files]
    with open(os.path.join("..", DATA_DIR, "test", "prompts_pp.txt"), "w") as f:
        f.write('\n'.join(sents))
'''