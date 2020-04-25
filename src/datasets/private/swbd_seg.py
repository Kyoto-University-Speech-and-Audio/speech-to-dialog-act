import tensorflow as tf
from ..base import BaseInputData


class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode, batch_size, dev=False):
        BaseInputData.__init__(self, hparams, mode, batch_size, dev)

        inputs = []
        with open(self.data_filename, "r") as f:
            headers = f.readline().strip().split('\t')
            for line in f.read().split('\n'):
                if self.mode != tf.estimator.ModeKeys.PREDICT:
                    if line.strip() == "": continue
                    input = { headers[i]: dat for i, dat in enumerate(line.strip().split('\t')) } 
                    if 'target' in input and self.hparams.use_sos_eos:
                        if hparams.use_seg_tag:
                            labels = input['target'].split(' ')
                            _labels = []
                            for l in labels:
                                if self.vocab[int(l)][:2] == '</':
                                    _labels.append(str(self.hparams.da_seg_index))
                                _labels.append(l)
                            input['target'] = ' '.join(_labels)
                        input['target'] = "%d %s %d" % (self.hparams.sos_index, input['target'], self.hparams.eos_index)
                    inputs.append(input)

            self.size = len(inputs)
            if self.hparams.sort_dataset:
                inputs.sort(key=lambda inp: inp['sound_len'])

            self.inputs = inputs

    def load_vocab(self, vocab_file):
        labels = [s.strip() for s in open(vocab_file, encoding=self.hparams.encoding)]
        vocab = {id: label for id, label in enumerate(labels)}
        vocab_size = len(vocab)
        if labels[-1][:2] != '</': # no specified DA tag
            self.hparams.eos_index = vocab_size
            self.hparams.sos_index = vocab_size + 1
            self.hparams.da_seg_index = vocab_size + 2
            vocab[vocab_size] = '<eos>'
            vocab[vocab_size + 1] = '<sos>'
            vocab[vocab_size + 2] = '</da>'
        elif self.hparams.use_seg_tag:
            self.hparams.eos_index = vocab_size
            self.hparams.sos_index = vocab_size + 1
            self.hparams.da_seg_index = self.vocab_size + 2
            vocab[vocab_size] = '<eos>'
            vocab[vocab_size + 1] = '<sos>'
            vocab[vocab_size + 2] = '</da>'
        else: # with specified DA tag
            self.hparams.eos_index = vocab_size
            self.hparams.sos_index = vocab_size + 1
            vocab[vocab_size] = '<eos>'
            vocab[vocab_size + 1] = '<sos>'
            self.tag_start = len(labels) - 43
            self.tag_end = len(labels) - 1
        return vocab

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
        
        self.dataset = src_tgt_dataset
        self.batched_dataset = self.get_batched_dataset(self.dataset)
        self.iterator = self.batched_dataset.make_initializable_iterator()


    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]
        
        sess.run(self.iterator.initializer, feed_dict={
            self.filenames: self.get_inputs_list(inputs, 'sound'),
            self.targets: self.get_inputs_list(inputs, 'target')
        })
