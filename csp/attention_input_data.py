import os
from python_speech_features import mfcc
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import attention_configs as configs

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 3

TGT_SOS_TOKEN = '<s>'
TGT_EOS_TOKEN = '</s>'
TGT_SOS_INDEX = 1
TGT_EOS_INDEX = 2

SRC_EOS_ID = 0

NUM_LABELS = 26 + 3

class BatchedInput():
    def __init__(self, batch_size=configs.BATCH_SIZE):
        search_path = os.path.join(configs.DATA_DIR, configs.FOLDER_WAV, '*.wav')
        self.input_files = gfile.Glob(search_path)
        self.target_files = [os.path.join(
            configs.DATA_DIR, configs.FOLDER_TXT,
            os.path.split(input_file)[-1].replace('.wav', '.txt')) for input_file in self.input_files]

        self.prepare_inputs()

        self.batch_size = batch_size

    def prepare_inputs(self):
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)

        spectrogram = contrib_audio.audio_spectrogram(
            wav_decoder.audio,
            window_size=int(configs.SAMPLE_RATE * configs.WINDOW_SIZE_MS / 1000),
            stride=int(configs.SAMPLE_RATE * configs.WINDOW_STRIDE_MS / 1000),
            magnitude_squared=True)
        inputs = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=configs.DCT_COEFFICIENT_COUNT)
        mean, var = tf.nn.moments(inputs, axes=[1])
        self.inputs_ = tf.divide(tf.subtract(inputs, mean), var)
        self.seq_len_ = tf.shape(self.inputs_)[1]

    def encode(self, s):
        original = ' '.join(str(s).strip().lower().split(' ')).replace('.', '').replace(',', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        # Adding blank label
        targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

        # Transform char into index
        targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                              for x in targets if x == SPACE_TOKEN or 'a' <= x <= 'z'] + [TGT_EOS_INDEX])
        return targets

    def get_data(self, sess):
        def gen():
            for i in range(configs.TRAINING_SIZE):
                input_dict = {
                    self.wav_filename_placeholder_: self.input_files[i],
                }
                inputs, seq_len = sess.run([self.inputs_, self.seq_len_], feed_dict=input_dict)
                # self.train_inputs.append(inputs[0])
                # self.train_seq_len.append(seq_len)
                yield (inputs[0], seq_len)

        tgt_dataset = tf.data.Dataset.from_tensor_slices(self.target_files)
        tgt_dataset = tgt_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).take(1))
        # tgt_dataset = tgt_dataset.map(lambda string: tf.string_split([string], delimiter="").values)
        tgt_dataset = tgt_dataset.map(lambda str: tf.py_func(self.encode, [str], tf.int64))
        tgt_dataset = tgt_dataset.map(lambda phones: (tf.cast(phones, tf.int32), tf.size(phones)))

        src_dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32))

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        self.batched_dataset = src_tgt_dataset.padded_batch(
            self.batch_size,
            padded_shapes=(([None, configs.DCT_COEFFICIENT_COUNT], []),
                           ([None], [])))

    def decode(self, d):
        str_decoded = ''.join([chr(x) for x in d + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        return str_decoded