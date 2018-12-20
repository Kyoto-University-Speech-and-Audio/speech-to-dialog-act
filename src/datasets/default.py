from .base import BaseInputData
import tensorflow as tf


class BatchedInput(BaseInputData):
    def __init__(self, hparams, mode, batch_size, dev=False):
        BaseInputData.__init__(self, hparams, mode, batch_size, dev)
        
        inputs = []
        with open(self.data_filename, "r") as f:
            headers = f.readline().strip().split(hparams.delimiter)
            for line in f.read().split('\n'):
                if line.strip() == "": continue

                input = { headers[i]: dat 
                        for i, dat in
                        enumerate(line.strip().split(hparams.delimiter,
                            len(headers) - 1))}
                inputs.append(input)

        self.size = len(inputs)
        self.inputs = inputs

    def init_dataset(self):
        src_dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        src_dataset = src_dataset.map(lambda filename: (filename, tf.py_func(self.load_input, [filename], tf.float32)))
        src_dataset = src_dataset.map(lambda filename, feat: (filename, feat, tf.shape(feat)[0]))

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            src_tgt_dataset = src_dataset
            self.batched_dataset = src_dataset.padded_batch(
                self.batch_size,
                padded_shapes=([], [None, self.hparams.num_features], []),
                padding_values=('', 0.0, 0)
            )
        else:
            tgt_dataset = tf.data.Dataset.from_tensor_slices(self.targets)
            tgt_dataset = tgt_dataset.map(
                lambda str: tf.cast(tf.py_func(self.extract_target_features, [str], tf.int64), tf.int32))
            tgt_dataset = tgt_dataset.map(lambda feat: (tf.cast(feat, tf.int32), tf.shape(feat)[0]))
            src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
            self.batched_dataset = self.get_batched_dataset(src_tgt_dataset)

        self.iterator = self.batched_dataset.make_initializable_iterator()

    def get_word(self, id):
        if self.vocab[id] == '': return ''
        if self.vocab[id][0] != '<':
            return self.vocab[id].split('+')[0]
        else:
            return ''

    def reset_iterator(self, sess, skip=0, shuffle=False, bucket_size=None):
        if shuffle: inputs = self.shuffle(self.inputs, bucket_size)
        else: inputs = self.inputs
        inputs = inputs[skip:]

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            sess.run(
                self.iterator.initializer,
                feed_dict={
                    self.filenames: self.get_inputs_list(inputs, 'sound')
                })
        else:
            sess.run(
                self.iterator.initializer,
                feed_dict={
                    self.filenames: self.get_inputs_list(inputs, 'sound'),
                    self.targets: self.get_inputs_list(inputs, 'target')
                })
