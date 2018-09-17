import random

import tensorflow as tf

from .base import BaseInputData
from ..utils import utils
import os
import numpy as np
from collections import namedtuple

class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode):
        BaseInputData.__init__(self, hparams, mode)

        inputs = []
        dlgs = {}
        self.size = 0
        # Utterance = namedtuple('Utterance', ['filename', 'first_utt', 'speaker', 'speaker_changed', 'target', 'start_frame'])
        for line in open(self.data_filename, "r"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                if line.strip() == "": continue
                filename, start_frame, target = line.strip().split('\t')
                target = "%d %s %d" % (self.hparams.sos_index, target, self.hparams.eos_index)
                start_frame = int(start_frame)
                speaker = 0 if os.path.basename(filename)[7] == 'A' else 1
                dlgid = os.path.basename(filename)[2:6]
                if dlgid in dlgs:
                    first_utt = True
                    for utt in dlgs[dlgid]:
                        if utt['speaker'] == speaker: first_utt = False
                    dlgs[dlgid].append(dict(
                        filename=filename,
                        first_utt=first_utt,
                        speaker=speaker,
                        speaker_changed=False,
                        target=target,
                        start_frame=start_frame))
                else: dlgs[dlgid] = [dict(
                    filename=filename,
                    first_utt=True,
                    speaker=speaker,
                    speaker_changed=False,
                    target=target,
                    start_frame=start_frame)]
                self.size += 1

        for dlgid in dlgs:
            dlgs[dlgid].sort(key=lambda it: it['start_frame'])

        for dlgid in dlgs:
            for i in range(1, len(dlgs[dlgid])):
                dlgs[dlgid][i]['speaker_changed'] = (dlgs[dlgid][i - 1]['speaker'] != dlgs[dlgid][i]['speaker'])

        # self.input_blank = ("", True, 0, "", 0)
        self.empty_utterance = dict(
            filename='',
            first_utt=False,
            speaker=0,
            speaker_changed=False,
            target="%d %d" % (self.hparams.sos_index, self.hparams.eos_index),
            start_frame=0
        )

        self.dlgs = dlgs
        # self.size = len(inputs)
        self.inputs = inputs

        self.is_first_utts = tf.placeholder(dtype=tf.bool, shape=[None])
        self.is_speaker_changed = tf.placeholder(dtype=tf.bool, shape=[None])
        self.speakers = tf.placeholder(dtype=tf.int32, shape=[None])

        dlgid = random.choice(list(dlgs.keys()))
        for utt in dlgs[dlgid]:
            print("%d: %s" % (utt['speaker'], self.decode([int(x) for x in utt['target'].split(' ')])))

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
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        src_is_first_utts_dataset = tf.data.Dataset.from_tensor_slices(self.is_first_utts)
        src_speakers_dataset = tf.data.Dataset.from_tensor_slices(self.speakers)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = src_dataset
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))

            src_tgt_dataset = tf.data.Dataset.zip(
                (src_dataset, src_is_first_utts_dataset, src_speakers_dataset, tgt_dataset))

        self.batched_dataset = src_tgt_dataset.padded_batch(
            self.batch_size,
            padded_shapes=(([], [None, self.hparams.num_features], []), [], [],
                           ([None], [])),
            padding_values=(('', 0.0, 0), False, -1, (self.hparams.eos_index, 0))
        )
        self.iterator = self.batched_dataset.make_initializable_iterator()

    def extract_target_features(self, str):
        return [[int(x) for x in str.decode('utf-8').split(' ')]]


    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        inputs = []
        dlgids = list(self.dlgs.keys())
        random.shuffle(dlgids)
        cur_dlgids = [dlgids[i] for i in range(self.batch_size)]
        pos = [0] * self.hparams.batch_size # position of current utterance for each batch
        cur_dlg = self.hparams.batch_size
        append_flg = True
        while append_flg:
            append_flg = False
            for i in range(self.batch_size):
                dlgid = cur_dlgids[i]
                if dlgid is not None:
                    inputs.append(self.dlgs[dlgid][pos[i]])
                    append_flg = True
                    pos[i] += 1
                    if pos[i] == len(self.dlgs[dlgid]):
                        cur_dlgids[i] = dlgids[cur_dlg] if cur_dlg < len(dlgids) else None 
                        pos[i] = 0
                        cur_dlg += 1
                else: inputs.append(self.empty_utterance)

        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: [i['filename'] for i in inputs],
            self.targets: [i['target'] for i in inputs],
            self.is_first_utts: [i['first_utt'] for i in inputs],
            self.speakers: [i['speaker'] for i in inputs],
            self.is_speaker_changed: [i['speaker_changed'] for i in inputs]
        })
