import os
from tensorflow.python.platform import gfile
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from csp.utils import wav_utils

DATA_DIR = os.path.join("data", "vivos")

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

class BatchedInput():
    def __init__(self, hparams, mode):
        # self.prepare_inputs()
        self.hparams = hparams

        self.mode_dir = "train" if mode == tf.estimator.ModeKeys.TRAIN else "test"

        search_path = os.path.join(DATA_DIR, self.mode_dir, "waves", '**', '*.wav')
        self.ls_input_files = gfile.Glob(search_path)
        self.ls_input_files.sort()

    def init_dataset(self):
        hparams = self.hparams
        self.batch_size = hparams.batch_size

        tgt_dataset = tf.data.TextLineDataset(os.path.join(DATA_DIR, self.mode_dir, "prompts_pp.txt"))
        # tgt_dataset = tgt_dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).take(1))
        # tgt_dataset = tgt_dataset.map(lambda string: tf.string_split([string], delimiter="").values)
        tgt_dataset = tgt_dataset.map(lambda str: tf.py_func(self.encode, [str], tf.int64))
        tgt_dataset = tgt_dataset.map(lambda phones: (tf.cast(phones, tf.int32), tf.size(phones)))

        # src_dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32))
        self.input_files = tf.placeholder(tf.string, shape=[None])
        src_dataset = tf.data.Dataset.from_tensor_slices(self.input_files)
        src_dataset = wav_utils.wav_to_features(src_dataset, hparams, 40)

        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        # src_tgt_dataset.shuffle(1000)

        if hparams.max_train > 0:
            src_tgt_dataset = src_tgt_dataset.take(hparams.max_train)

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
    num_classes = len(CHARS) + FIRST_INDEX

    def size(self): return len(self.ls_input_files)

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
        str_decoded = ""
        for x in d:
            if len(CHARS) + FIRST_INDEX > x >= FIRST_INDEX: str_decoded += CHARS[x - FIRST_INDEX]
            elif x == 0: str_decoded += " "
            else: break
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