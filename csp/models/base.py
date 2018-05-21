import tensorflow as tf
from ..utils import model_utils

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class BaseModel(object):

    def __init__(self, hparams, mode, iterator):
        self.iterator = iterator
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

        self.global_step = tf.Variable(0, trainable=False)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.inputs, self.input_seq_len = self.iterator.get_next()
            self.target_labels, self.target_seq_len = None, None
        else:
            ((self.inputs, self.input_seq_len), (self.target_labels, self.target_seq_len)) = \
                self.iterator.get_next()

        self.batch_size = tf.shape(self.input_seq_len)[0]
        self._build_graph()
        params = tf.trainable_variables()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            self.learning_rate = self._get_learning_rate_warmup(hparams)
            self.learning_rate = self._get_learning_rate_decay(hparams)

            if hparams.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif hparams.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif hparams.optimizer == "momentum":
                opt = tf.train.MomentumOptimizer(self.learning_rate,
                                                            0.9).minimize(self.loss)

            gradients = tf.gradients(
                self.loss,
                params,
                colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
                gradients, max_gradient_norm=MAX_GRADIENT_NORM)
            self.grad_norm = grad_norm

            self.update = opt.apply_gradients(zip(clipped_grads, params), global_step=self.global_step)

            self.train_summary = tf.summary.merge([
                tf.summary.scalar("learning_rate", self.learning_rate),
                tf.summary.scalar('train_loss', self.loss),
                tf.summary.scalar('label_error_rate', self.ler)
            ] + grad_norm_summary)

            if self.infer_mode:
                self.infer_summary = self._get_infer_summary(hparams)

            print("# Trainable variables")
            for param in params:
                print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                                  param.op.device))
        self.saver = tf.train.Saver(tf.global_variables())

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
                tf.to_float(WARMUP_STEPS - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % WARMUP_SCHEME)

        return tf.cond(
            self.global_step < WARMUP_STEPS,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self, hparams):
        """Get learning rate decay."""
        if DECAY_SCHEME in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if DECAY_SCHEME == "luong5":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 5
            elif DECAY_SCHEME == "luong10":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 10
            elif DECAY_SCHEME == "luong234":
                start_decay_step = int(hparams.num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not DECAY_SCHEME:  # no decay
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif DECAY_SCHEME:
            raise ValueError("Unknown decay scheme %s" % DECAY_SCHEME)
        print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                        "decay_factor %g" % (DECAY_SCHEME,
                                             start_decay_step,
                                             decay_steps,
                                             decay_factor))
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _get_infer_summary(self, hparams):
        return tf.no_op()
