import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils
import numpy as np

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode, batch_size, dev=False):
        BaseInputData.__init__(self, hparams, mode, batch_size, dev)

        inputs = []
        with open(self.data_filename, "r") as f:
            lines = f.read().split('\n')
            headers = lines[0].strip().split('\t')
            for line in lines[1:]:
                if line.strip() == "": continue
                input = { headers[i]: dat for i, dat in enumerate(line.strip().split('\t')) }
                if 'target' in input: input['target'] = "%d %s %d" % (self.hparams.sos_index, input['target'], self.hparams.eos_index)
                #else: print(line)
                inputs.append(input)
        
        if hparams.predicted_train_data is not None:
            with open(hparams.predicted_train_data if mode == "train" else \
                    (hparams.predicted_dev_data if dev else hparams.predicted_test_data), "r") as f:
                for _id, line in enumerate(f.read().split('\n')):
                    if line.strip() == "": continue
                    fields = line.split('\t')
                    if len(fields) != 3 or _id >= len(inputs):
                        pass
                    else:
                        inputs[_id]['predicted_text'] = fields[1] if fields[1].strip() != '' else '0'
                        inputs[_id]['context_filename'] = fields[2]
            self.predicted_texts = tf.placeholder(dtype=tf.string)
            self.context_filenames = tf.placeholder(dtype=tf.string)

        self.size = len(inputs)
        self.inputs = inputs

        #for inp in inputs:
        #    if 'target' not in inp: print(inp)

        self.dlg_ids = tf.placeholder(dtype=tf.string)
        self.dialog_acts = tf.placeholder(dtype=tf.int32)

    def load_vocab(self, vocab_file):
        labels = [s.strip() for s in open(vocab_file, encoding=self.hparams.encoding)]
        self.decoder_map = {id: label for id, label in enumerate(labels)}
        self.num_classes = len(labels) + 2
        self.hparams.eos_index = self.num_classes - 2
        self.hparams.sos_index = self.num_classes - 1
        self.decoder_map[self.num_classes - 2] = '<eos>'
        self.decoder_map[self.num_classes - 1] = '<sos>'

    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (tf.py_func(self.load_input, [filename], tf.float32)))
        #src_dataset = src_dataset.map(lambda filename: (tf.constant([[0.0] * 120])))
        src_dataset = src_dataset.map(lambda feat: (feat, tf.shape(feat)[0]))

        tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
        tgt_dataset = tgt_dataset.map(
            lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
        tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

        if self.hparams.predicted_train_data is not None:
            pt_dataset = tf.data.Dataset.from_tensor_slices(self.predicted_texts)
            pt_dataset = pt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            pt_dataset = pt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            af_dataset = tf.data.Dataset.from_tensor_slices(self.context_filenames)
            #af_dataset = af_dataset.map(lambda filename: (tf.py_func(self.load_input, [filename], tf.float32)))
            af_dataset = af_dataset.map(lambda filename: tf.random_normal([50, 512]))

            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.dlg_ids),
                src_dataset,
                tgt_dataset,
                pt_dataset,
                af_dataset,
                tf.data.Dataset.from_tensor_slices(self.dialog_acts)
            ))

            self.batched_dataset = dataset.padded_batch(
                self.batch_size,
                padded_shapes=([],
                               ([None, self.hparams.num_features], []),
                               ([None], []),
                               ([None], []),
                               [None, 512],
                               []),
                padding_values=('',
                    (0.0, 0),
                    (self.hparams.eos_index, 0),
                    (self.hparams.eos_index, 0),
                    0.0,
                    0)
            )
        else:
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.dlg_ids),
                src_dataset,
                tgt_dataset,
                tf.data.Dataset.from_tensor_slices(self.dialog_acts)
            ))

            self.batched_dataset = dataset.padded_batch(
                self.batch_size,
                padded_shapes=([],
                               ([None, self.hparams.num_features], []),
                               ([None], []),
                               []),
                padding_values=('',
                                (0.0, 0),
                                (self.hparams.eos_index, 0),
                                0)
            )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, s):
        return [[int(x) for x in s.decode('utf-8').split(' ')]]

    def get_inputs_list(self, field):
        return [inp[field] for inp in self.inputs]

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]
        if self.hparams.predicted_train_data is not None:
            sess.run(self.iterator.initializer, feed_dict={
                self.dlg_ids: self.get_inputs_list('dialog_id'),
                self.filenames: self.get_inputs_list('sound'),
                self.targets: self.get_inputs_list('target'),
                self.predicted_texts: self.get_inputs_list('predicted_text'),
                self.context_filenames: self.get_inputs_list('context_filename'),
                self.dialog_acts: self.get_inputs_list('dialog_act')
            })
        else:
            sess.run(self.iterator.initializer, feed_dict={
                self.dlg_ids: self.get_inputs_list('dialog_id'),
                self.filenames: self.get_inputs_list('sound'),
                self.targets: self.get_inputs_list('target'),
                self.dialog_acts: self.get_inputs_list('dialog_act')
            })

    def decode(self, d, id=None):
        if type(d) == np.ndarray:
            return super().decode(d)
        else:
            return [str(d)]
