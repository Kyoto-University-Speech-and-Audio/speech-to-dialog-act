import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode, batch_size, dev=False):
        BaseInputData.__init__(self, hparams, mode, batch_size, dev)
        
        inputs = []
        for line in open(self.data_filename, "r"):
            if line.strip() == "": continue
            filename, target = line.strip().split(' ', 1)
            inputs.append({
                'filename': filename, 
                'target': target
            })

        self.size = len(inputs)
        self.inputs = inputs

    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = src_dataset
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        self.batched_dataset = self.get_batched_dataset(src_tgt_dataset)
        self.iterator = self.batched_dataset.make_initializable_iterator()

    def get_word(self, id):
        if self.hparams.input_unit == "word":
            return self.decoder_map[id].split('+')[0]
        else:
            return self.decoder_map[id]
    
    def get_inputs_list(self, field):
        return [inp[field] for inp in self.inputs]

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: self.get_inputs_list("filename"),
            self.targets: self.get_inputs_list("target")
        })
