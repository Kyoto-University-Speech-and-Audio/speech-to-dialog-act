import os
from python_speech_features import mfcc
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import configs

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

class AudioProcessor():
    def __init__(self):
        search_path = os.path.join(configs.DATA_DIR, configs.FOLDER_WAV, '*.wav')
        self.audio_files = gfile.Glob(search_path)
        self.target_files = [os.path.join(
            configs.DATA_DIR, configs.FOLDER_TXT,
            os.path.split(audio_file)[-1].replace('.wav', '.txt')) for audio_file in self.audio_files]

        self.prepare_inputs()
        self.prepare_targets()

        self.train_inputs = []
        self.train_seq_len = []
        self.train_targets = []

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

    def encode(self, str):
        original = ' '.join(str.strip().lower().split(' ')).replace('.', '').replace(',', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        # Adding blank label
        targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

        # Transform char into index
        targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                              for x in targets if x == SPACE_TOKEN or 'a' <= x <= 'z'])
        return targets

    def prepare_targets(self):
        self.targets_ = tf.placeholder(tf.int8, shape=[None])

    def get_data(self, sess):
        for i in range(configs.TRAINING_SIZE):
            f = open(self.target_files[i], 'r')
            text = f.readlines()[-1]
            f.close()
            input_dict = {
                self.wav_filename_placeholder_: self.audio_files[i],
            }
            inputs, seq_len = sess.run([self.inputs_, self.seq_len_], feed_dict=input_dict)
            self.train_inputs.append(inputs[0])
            self.train_seq_len.append(seq_len)
            self.train_targets.append(self.encode(text))

    def next_train_batch(self, batch, batch_size):
        batch_end = batch + batch_size
        return self.train_inputs[batch:batch_end], \
               self.train_seq_len[batch:batch_end], \
               self.sparse_tuple_from(self.train_targets[batch:batch_end])

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

        return tf.SparseTensorValue(indices, values, shape)

    def decode(self, d):
        str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        return str_decoded