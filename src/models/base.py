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
                 batched_input,
                 **kwargs):
        """

        :param hparams:
        :param mode:
        :param batched_input: instance of BatchedInput
        :param kwargs:
        :return:
        """
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

        self._batched_input = batched_input
        self._assign_input()
        self.batch_size = tf.shape(self.inputs)[0]

        if self.train_mode:
            self._build_train_model(hparams, **kwargs)
        elif self.eval_mode:
            self._build_eval_model()
        elif self.infer_mode:
            self._build_infer_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_train_model(self, hparams, **kwargs):
        self.loss = self._build_graph()

    def _build_eval_model(self):
        self.loss = self._build_graph()
        self.summary = tf.summary.merge([
            tf.summary.scalar('eval_loss', self.loss),
        ])

    def _build_infer_model(self):
        self.target_labels, self.target_seq_len = None, None
        self._build_graph()
        # self.summary = self._get_attention_summary()

    @property
    def iterator(self):
        return self._batched_input.iterator

    def _assign_input(self):
        """
        Override this to implement your input variables
        :return:
        """
        if self.eval_mode or self.train_mode:
            ((self.input_filenames, self.inputs, self.input_seq_len), (self.target_labels, self.target_seq_len)) = \
                self.iterator.get_next()
        else:
            self.input_filenames, self.inputs, self.input_seq_len = self.iterator.get_next()

    def _build_graph(self):
        """
        Override this to build your model
        :return:
        """
        pass

    @classmethod
    def load(cls, sess, ckpt, flags):
        """
        Load with --transfer. Override this to implement your own transfer learning.
        :param sess:
        :param ckpt:
        :param flags:
        :return:
        """
        saver_variables = tf.global_variables()

        var_list = {var.op.name: var for var in saver_variables}
        del var_list['Variable_1']

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    @classmethod
    def ignore_save_variables(cls):
        """
        List of variables that won't be saved in checkpoint
        :return:
        """
        return []

    @classmethod
    def trainable_variables(cls):
        """
        List of variables that will be trained. Exclude some parameters if you want to freeze certain parts of the model
        :return:
        """
        return tf.trainable_variables()
