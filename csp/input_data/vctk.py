import os
from tensorflow.python.platform import gfile
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

DATA_DIR = os.path.join("..", "data", "VCTK-Corpus")

CHARS = "abcdefghijklmnopqrstuvwxyz"

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

TRAINING_SIZE = -1
TRAINING_START = 1

WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 10.0

SAMPLE_RATE = 16000

class BatchedInput():
    def __init__(self, mode, batch_size):
        # self.prepare_inputs()
        self.batch_size = batch_size

        wav_search_path = os.path.join(DATA_DIR, "wav48", '**', '*.wav')
        ls_input_files = gfile.Glob(wav_search_path)

        txt_search_path = os.path.join(DATA_DIR, "txt", '**', '*.txt')
        ls_target_files = gfile.Glob(txt_search_path)

        ifiles = set([os.path.basename(os.path.splitext(path)[0]) for path in ls_input_files])
        tfiles = set([os.path.basename(os.path.splitext(path)[0]) for path in ls_target_files])

        files = ifiles - (ifiles - tfiles)
        self.ls_input_files = [os.path.join(DATA_DIR, "wav48", fn.split('_')[0], fn + '.wav') for fn in files]
        self.ls_target_files = [os.path.join(DATA_DIR, "txt", fn.split('_')[0], fn + '.txt') for fn in files]

        self.target_files = tf.placeholder(tf.string, shape=[None])
        tgt_dataset = tf.data.Dataset.from_tensor_slices(self.target_files).skip(TRAINING_START)
        tgt_dataset = tgt_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).take(1))
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

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        src_tgt_dataset.shuffle(1000)

        if TRAINING_SIZE > 0: src_tgt_dataset.take(TRAINING_SIZE)

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

    def encode(self, s):
        s = s.decode('utf-8')
        original = ' '.join(str(s).strip().lower().split(' ')).replace('.', '').replace(',', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        # Adding blank label
        targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

        # Transform char into index
        targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else FIRST_INDEX + CHARS.index(x)
                              for x in targets if x == SPACE_TOKEN or x in CHARS])
        return targets

    def reset_iterator(self, sess):
        sess.run(self.iterator.initializer, feed_dict={
            self.input_files: self.ls_input_files,
            self.target_files: self.ls_target_files
        })

    def decode(self, d):
        str_decoded = ''.join([CHARS[x - FIRST_INDEX] if x >= FIRST_INDEX else ' ' for x in d])
        return str_decoded