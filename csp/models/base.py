import tensorflow as tf
from ..utils import model_utils, utils

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class BaseModel(object):
    def __call__(self,
                 hparams,
                 mode,
                 iterator,
                 **kwargs):
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

        self.iterator = iterator
        # self.batch_size = hparams.batch_size

        self.batch_size = tf.shape(self.input_seq_len)[0]
        if self.train_mode:
            self._build_train_model(hparams, **kwargs)
        elif self.eval_mode:
            self._build_eval_model()
        elif self.infer_mode:
            self._build_infer_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def _get_trainable_params(self, **kwargs):
        if "trainable_scope" in kwargs:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=kwargs['trainable_scope'])
        else:
            return tf.trainable_variables()

    def _build_train_model(self, hparams, **kwargs):
        self.loss = self._build_graph()
        self.params = self._get_trainable_params(**kwargs)

    def _build_eval_model(self):
        # self.batch_size = tf.shape(self.input_seq_len)[0]
        self.loss = self._build_graph()
        self.summary = tf.summary.merge([
            tf.summary.scalar('eval_loss', self.loss),
        ])

    def _build_infer_model(self):
        # self.batch_size = tf.shape(self.input_seq_len)[0]
        self.target_labels, self.target_seq_len = None, None
        self._build_graph()
        # self.summary = self._get_attention_summary()

    @property
    def iterator(self):
        return self._iterator

    @iterator.setter
    def iterator(self, value):
        self._iterator = value
        self._assign_input()

    def _assign_input(self):
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len), (self.target_labels, self.target_seq_len)) = \
                self._iterator.get_next()
        else:
            self.input_filenames, self.inputs, self.input_seq_len = self._iterator.get_next()


    def _build_graph(self):
        pass

    def load(self, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        #if flags.load_ignore_scope:
        #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FLAGS.load_ignore_scope):
        #        saver_variables.remove(var)

        var_list = {var.op.name: var for var in saver_variables}

        var_map = {
            # "decoder/decoder_emb_layer/kernel": "decoder/dense/kernel",
            # "decoder/decoder_emb_layer/bias": "decoder/dense/bias",
            # "decoder/decoder_emb_layer/bias/Adam": "decoder/dense/bias/Adam",
            # "decoder/decoder_emb_layer/bias/Adam_1": "decoder/dense/bias/Adam_1",
            # "decoder/decoder_emb_layer/kernel/Adam": "decoder/dense/kernel/Adam",
            # "decoder/decoder_emb_layer/kernel/Adam_1": "decoder/dense/kernel/Adam_1",
        }

        # fine-tuning for context
        # print(var_list)
        # del var_list["decoder/attention_wrapper/basic_lstm_cell/kernel"]

        for it in var_map:
            var_list[var_map[it]] = var_list[it]
        # del var_list[it]

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)


