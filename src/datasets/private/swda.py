import random

import tensorflow as tf

from datasets.base import BaseInputData
from utils import utils
import numpy as np

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode, batch_size, dev=False):
        BaseInputData.__init__(self, hparams, mode, batch_size, dev)

        inputs = self.inputs
        #with open("data/swbd/model_outputs/5_beam.csv", "r") as f:
        #    for _id, line in enumerate(f.read().split('\n')):
        #        if line.strip() == "": continue
        #        inputs[_id]['target'] = "%d %s %d" % (self.hparams.sos_index,
        #                line.split('\t')[1], self.hparams.eos_index)
        
        if hparams.predicted_test_data is not None:
            with open(hparams.predicted_train_data if mode == "train" else \
                    (hparams.predicted_dev_data if dev else hparams.predicted_test_data), "r") as f:
                for _id, line in enumerate(f.read().split('\n')):
                    if line.strip() == "": continue
                    fields = line.split('\t')
                    if _id >= len(inputs):
                        pass
                    else:
                        inputs[_id]['predicted_text'] = fields[1] if fields[1].strip() != '' else '0'
                        if len(fields) > 2: inputs[_id]['context_filename'] = fields[2]
                        if hparams.get('forced_decoding', False):
                            inputs[_id]['target'] = "%d %s %d" % (self.hparams.sos_index,
                                    inputs[_id]['predicted_text'], self.hparams.eos_index)
            self.predicted_texts = tf.placeholder(dtype=tf.string)
            self.context_filenames = tf.placeholder(dtype=tf.string)

        #for inp in inputs:
        #    if 'target' not in inp: print(inp)

        self.dlg_ids = tf.placeholder(dtype=tf.string)
        self.dialog_acts = tf.placeholder(dtype=tf.int32)

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
        return [[int('0' + x) for x in s.decode('utf-8').split(' ')]]

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
            return super().decode(d, id)
        else:
            return [str(d)]
    
    def decode_da(self, d, id=None):
        return [str(d)]
        ret = []
        da_tags = ['fo_o_fw_"_by_bc', 'qw', 'h', 'sd', 'sv', 'b', 'x', '%', '+', 'qy', 'qrr', 'na', 'bk', 'ba', 'ny', '^q', 'aa', 'nn', 'fc', 'ad', 'qo', 'qh', 'no', 'ng', '^2', 'bh', 'qy^d', 'br', 'b^m', '^h', 'bf', 'fa', 'oo_co_cc', 'ar', 'bd', 't1', 'arp_nd', 't3', 'ft', '^g', 'qw^d', 'fp', 'aap_am']
        da_tags = ['-'] + ["</%s>" % t for t in da_tags] + ['-']
        for c in d:
            #if c == self.hparams.eot_index: break
            ret.append(da_tags[c])
        return ret
