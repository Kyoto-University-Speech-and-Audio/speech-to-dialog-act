import random

import tensorflow as tf

from ..base import BaseInputData
from utils import utils
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
                if 'target' in input and self.hparams.use_sos_eos: 
                    input['target'] = "%d %s %d" % (self.hparams.sos_index, input['target'], self.hparams.eos_index)
                inputs.append(input)
        
        self.extra_data_path = hparams.predicted_train_data if mode == "train" else \
                (hparams.predicted_dev_data if dev else hparams.predicted_test_data)
        self.extra_data = self.extra_data_path is not None

        if self.extra_data:
            with open(self.extra_data_path, "r") as f:
                for _id, line in enumerate(f.read().split('\n')):
                    if line.strip() == "": continue
                    fields = line.split('\t')
                    if len(fields) != 3 or _id >= len(inputs):
                        pass
                    else:
                        inputs[_id]['predicted_text'] = fields[1] if fields[1].strip() != '' else '0'
                        if self.hparams.use_sos_eos:
                            inputs[_id]['predicted_text'] = "%d %s %d" % (self.hparams.sos_index, inputs[_id]['predicted_text'], self.hparams.eos_index)
                        inputs[_id]['context_filename'] = fields[2]

            assert all('predicted_text' in inputs[_id] for _id in range(len(inputs)))
            self.predicted_texts = tf.placeholder(dtype=tf.string)
            self.context_filenames = tf.placeholder(dtype=tf.string)

        self.size = len(inputs)
        self.inputs = inputs

        #for inp in inputs:
        #    if 'target' not in inp: print(inp)

        self.dlg_ids = tf.placeholder(dtype=tf.string)
        self.dialog_acts = tf.placeholder(dtype=tf.string)

    def load_vocab(self, vocab_file):
        labels = [s.strip() for s in open(vocab_file, encoding=self.hparams.encoding)]
        vocab = {id: label for id, label in enumerate(labels)}
        if self.hparams.use_sos_eos:
            vocab_size = len(labels) + 3
            self.hparams.eos_index = vocab_size - 3
            self.hparams.sos_index = vocab_size - 2
            self.hparams.da_seg_index = vocab_size - 1
            vocab[vocab_size - 3] = '<eos>'
            vocab[vocab_size - 2] = '<sos>'
            vocab[vocab_size - 1] = '</da>'
        else:
            self.hparams.eos_index = 0
            #vocab[len(vocab)] = '<eos>'
            vocab_size = len(labels)
        
        self.hparams.eot_index = 0
        if self.hparams.model == "da_seg":
            self.hparams.eot_index = 0

        return vocab

    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        if self.hparams.load_voice:
            src_dataset = src_dataset.map(lambda filename: (tf.py_func(self.load_input, [filename], tf.float32)))
        else:
            src_dataset = src_dataset.map(lambda filename: (tf.constant([[0.0] * 120])))
        src_dataset = src_dataset.map(lambda feat: (feat, tf.shape(feat)[0]))

        tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
        tgt_dataset = tgt_dataset.map(
            lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
        tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

        da_dataset = tf.data.Dataset.from_tensor_slices(self.dialog_acts)
        da_dataset = da_dataset.map(
            lambda str: tf.cast(tf.py_func(self.extract_da_features, [str], tf.int64), tf.int32))
        da_dataset = da_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

        if self.extra_data:
            pt_dataset = tf.data.Dataset.from_tensor_slices(self.predicted_texts)
            pt_dataset = pt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            pt_dataset = pt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            #af_dataset = tf.data.Dataset.from_tensor_slices(self.context_filenames)
            #af_dataset = af_dataset.map(lambda filename: (tf.py_func(self.load_input, [filename], tf.float32)))
            #af_dataset = af_dataset.map(lambda filename: tf.random_normal([50, 512]))

            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.dlg_ids),
                src_dataset,
                tgt_dataset,
                pt_dataset,
                #af_dataset,
                da_dataset
            ))

            self.batched_dataset = dataset.padded_batch(
                self.batch_size,
                padded_shapes=([],
                               ([None, self.hparams.num_features], []),
                               ([None], []),
                               ([None], []),
                               #[None, 512],
                               ([None], [])),
                padding_values=('',
                    (0.0, 0),
                    (self.hparams.eos_index, 0),
                    (self.hparams.eos_index, 0),
                    #0.0,
                    #0,
                    (self.hparams.eot_index, 0))
            )
        else:
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(self.dlg_ids),
                src_dataset,
                tgt_dataset,
                da_dataset
            ))

            self.batched_dataset = dataset.padded_batch(
                self.batch_size,
                padded_shapes=([],
                               ([None, self.hparams.num_features], []),
                               ([None], []),
                               ([None], [])),
                padding_values=('',
                                (0.0, 0),
                                (self.hparams.eos_index, 0),
                                (self.hparams.eot_index, 0))
            )

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, s):
        return [[int(x) for x in s.decode('utf-8').split(' ')]]

    def extract_da_features(self, s):
        return [[int(x) for x in s.decode('utf-8').split(',')]]

    def get_inputs_list(self, field):
        return [inp[field] if field in inp else '' for inp in self.inputs]

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]
        if self.extra_data:
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

    def decode_da(self, d, id=None):
        ret = []
        da_tags = ['fo_o_fw_"_by_bc', 'qw', 'h', 'sd', 'sv', 'b', 'x', '%', '+', 'qy', 'qrr', 'na', 'bk', 'ba', 'ny', '^q', 'aa', 'nn', 'fc', 'ad', 'qo', 'qh', 'no', 'ng', '^2', 'bh', 'qy^d', 'br', 'b^m', '^h', 'bf', 'fa', 'oo_co_cc', 'ar', 'bd', 't1', 'arp_nd', 't3', 'ft', '^g', 'qw^d', 'fp', 'aap_am']
        da_tags = ['-'] + ["</%s>" % t for t in da_tags] + ['-']
        for c in d:
            #if c == self.hparams.eot_index: break
            ret.append(da_tags[c])
        return ret

    def decode(self, d, id=None):
        return super().decode(d, id)
