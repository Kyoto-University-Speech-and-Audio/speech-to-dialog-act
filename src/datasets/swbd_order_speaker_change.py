import random

import tensorflow as tf

from .swbd_order import BatchedInput as BaseInputData
from ..utils import utils
import os
import numpy as np
from collections import namedtuple

class BatchedInput(BaseInputData):
    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        src_speaker_changed_dataset = tf.data.Dataset.from_tensor_slices(self.is_speaker_changed)

        dataset = tf.data.Dataset.zip(src_dataset, src_speaker_changed_dataset)

        self.batched_dataset = src_dataset.padded_batch(
            self.batch_size,
            padded_shapes=([None, self.hparams.num_features], []),
            padding_values=((0.0, 0), False)
        )
        self.iterator = self.batched_dataset.make_initializable_iterator()