import os
from tensorflow.python.platform import gfile
import numpy as np

from csp.utils import wav_utils

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

DATA_DIR = os.path.join("data", "VCTK-Corpus")

CHARS = "abcdefghijklmnopqrstuvwxyz"

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = 1

SRC_EOS_ID = 0

DCT_COEFFICIENT_COUNT = 40

class BatchedInput():
    def __init__(self, hparams, mode):
        # self.prepare_inputs()
        # hparams.sample_rate = 96000
        self.batch_size = hparams.batch_size

        wav_search_path = os.path.join(DATA_DIR, "wav48", '**', '*.wav')
        ls_input_files = gfile.Glob(wav_search_path)

        txt_search_path = os.path.join(DATA_DIR, "txt", '**', '*.txt')
        ls_target_files = gfile.Glob(txt_search_path)

        ifiles = set([os.path.basename(os.path.splitext(path)[0]) for path in ls_input_files])
        tfiles = set([os.path.basename(os.path.splitext(path)[0]) for path in ls_target_files])

        files = list(ifiles - (ifiles - tfiles))
        self.ls_input_files = [os.path.join(DATA_DIR, "wav48", fn.split('_')[0], fn + '.wav') for fn in files]
        self.ls_target_files = [os.path.join(DATA_DIR, "txt", fn.split('_')[0], fn + '.txt') for fn in files]

        self.target_files = tf.placeholder(tf.string, shape=[None])
        tgt_dataset = tf.data.Dataset.from_tensor_slices(self.target_files)
        tgt_dataset = tgt_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).take(1))
        # tgt_dataset = tgt_dataset.map(lambda string: tf.string_split([string], delimiter="").values)
        tgt_dataset = tgt_dataset.map(lambda str: tf.py_func(self.encode, [str], tf.int64))
        tgt_dataset = tgt_dataset.map(lambda phones: (tf.cast(phones, tf.int32), tf.size(phones)))

        # src_dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32))
        self.input_files = tf.placeholder(tf.string, shape=[None])
        src_dataset = tf.data.Dataset.from_tensor_slices(self.input_files)
        src_dataset = wav_utils.wav_to_features(src_dataset, hparams, 40)

        if hparams.max_train > 0:
            src_dataset.take(hparams.max_train)
            tgt_dataset.take(hparams.max_train)

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        src_tgt_dataset.shuffle(1000)

        self.batched_dataset = src_tgt_dataset

        def batching_func(x):
            return x.padded_batch(
                self.batch_size,
                padded_shapes=(([None, DCT_COEFFICIENT_COUNT], []),
                               ([None], [])))

        if hparams.num_buckets > 1:
            def key_func(src, tgt):
                bucket_width = 10

                # Bucket sentence pairs by the length of their source sentence and target
                # sentence.
                bucket_id = tf.maximum(src[1] // bucket_width, tgt[1] // bucket_width)
                return tf.to_int64(tf.minimum(hparams.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            self.batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=hparams.batch_size))

        self.iterator = self.batched_dataset.make_initializable_iterator()

    num_features = DCT_COEFFICIENT_COUNT
    num_classes = len(CHARS) + FIRST_INDEX + 1

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