import tensorflow as tf
from ...utils import model_utils, utils

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class Trainer(object):
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

    def init(self, sess):
        # self.processed_inputs_count = sess.run(self._processed_inputs_count)
        pass

    def __call__(self,
                 model_fn,
                 batched_input,
                 **kwargs):
        self._global_step = tf.Variable(0, trainable=False)

        self._batched_input = batched_input
        self.iterator = batched_input.iterator
        # self.batch_size = hparams.batch_size

        self.batch_size = self.hparams.batch_size
        if self.train_mode:
            self._build_train_model(model_fn, **kwargs)
        elif self.eval_mode:
            self._build_eval_model(model_fn)
        elif self.infer_mode:
            self._build_infer_model()

        print("# Variables")
        for param in tf.global_variables():
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

        if self.train_mode:
            print("# Trainable variables")
            for param in self.params:
                print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

        self.saver = tf.train.Saver(tf.global_variables())

    def _get_trainable_params(self, **kwargs):
        if "trainable_scope" in kwargs:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=kwargs['trainable_scope'])
        else:
            return tf.trainable_variables()

    def _build_train_model(self, model_fn, **kwargs):
        self.learning_rate = tf.constant(self.hparams.learning_rate)
        self.learning_rate = self._get_learning_rate_warmup(self.hparams)
        self.learning_rate = self._get_learning_rate_decay()

        model = model_fn()
        opt = utils.get_optimizer(self.hparams, self.learning_rate)
        self.loss = model.loss
        self.params = self._get_trainable_params(**kwargs)

        gradients = opt.compute_gradients(
            self.loss,
            colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

        #clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
        #    gradients, max_gradient_norm=MAX_GRADIENT_NORM)
        #self.grad_norm = grad_norm

        self.update = opt.apply_gradients(gradients, global_step=self.global_step)

        self.train_summary = tf.summary.merge([
                                                  tf.summary.scalar('train_loss', self.loss),
                                                  tf.summary.scalar("learning_rate", self.learning_rate),
                                              ])

    def _build_eval_model(self, model_fn):
        self.model = model_fn()
        self.loss = self.model.loss
        #self.summary = tf.summary.merge([
        #    tf.summary.scalar('eval_loss', self.loss),
        #])

    def _build_infer_model(self):
        # self.batch_size = tf.shape(self.input_seq_len)[0]
        self.target_labels, self.target_seq_len = None, None
        self._build_graph()
        # self.summary = self._get_attention_summary()

    def eval(self, sess):
        input_filenames, target_labels, sample_ids, loss, summary = sess.run([
            self.model.input_filenames,
            self.model.target_labels,
            self.model.sample_id,
            self.model.loss, tf.no_op(),
        ])
        return input_filenames, target_labels, sample_ids, summary

    @property
    def iterator(self):
        return self._iterator

    @iterator.setter
    def iterator(self, value):
        self._iterator = value

    def _get_learning_rate_warmup(self, hparams):
        """Get learning rate warmup."""
        print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                        (hparams.learning_rate, WARMUP_STEPS, WARMUP_SCHEME))

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if WARMUP_SCHEME == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / WARMUP_STEPS)
            inv_decay = warmup_factor ** (
                tf.to_float(WARMUP_STEPS - self._global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % WARMUP_SCHEME)

        return tf.cond(
            self._global_step < WARMUP_STEPS,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self):
        start_decay_step = 200000
        decay_steps = 20000
        decay_rate = 0.5

        return tf.cond(
            self._global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self._global_step - start_decay_step),
                decay_steps, decay_rate, staircase=True),
            name="learning_rate_decay_cond")

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

    @property
    def global_step(self):
        return 0
        return self.processed_inputs_count / self.hparams.batch_size

    @property
    def data_size(self):
        return self._batched_input.size

    @property
    def epoch(self):
        return self.processed_inputs_count // self.data_size + 1

    @property
    def epoch_exact(self):
        return 0
        return self.processed_inputs_count / self.data_size

    @property
    def epoch_progress(self):
        return (self.processed_inputs_count % self.data_size) // self.hparams.batch_size