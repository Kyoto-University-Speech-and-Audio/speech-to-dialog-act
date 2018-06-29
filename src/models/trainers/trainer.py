import tensorflow as tf
from ...utils import model_utils, utils, ops_utils

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class Trainer(object):
    def __init__(self, hparams, Model, BatchedInput, mode):
        self.hparams = hparams
        self.mode = mode
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT
        self.Model = Model
        self.BatchedInput = BatchedInput
        self.batch_size = self.hparams.eval_batch_size if self.eval_mode else self.hparams.batch_size

        batched_input_train = BatchedInput(self.hparams, tf.estimator.ModeKeys.TRAIN)
        batched_input_eval = BatchedInput(self.hparams, tf.estimator.ModeKeys.EVAL)
        batched_input_train.init_dataset()
        batched_input_eval.init_dataset()
        self.hparams.num_classes = batched_input_train.num_classes

        self._processed_inputs_count = tf.Variable(0, trainable=False)
        self.processed_inputs_count = 0
        self.increment_inputs_count = tf.assign(
            self._processed_inputs_count,
            self._processed_inputs_count + (self.batch_size))

        self._global_step = tf.Variable(0, trainable=False)
        self._batched_input_train = batched_input_train
        self._batched_input_eval = batched_input_eval

    def init(self, sess):
        self.processed_inputs_count = sess.run(self._processed_inputs_count)
        self.reset_train_iterator(sess)
        self._batched_input_eval.reset_iterator(sess)

    def build_model(self, eval=False):
        if self.train_mode:
            with tf.variable_scope(tf.get_variable_scope()):
                self.learning_rate = tf.constant(self.hparams.learning_rate)
                self.learning_rate = self._get_learning_rate_warmup(self.hparams)
                self.learning_rate = self._get_learning_rate_decay()

                model = self.Model()
                model(
                    self.hparams,
                    tf.estimator.ModeKeys.TRAIN,
                    self._batched_input_train.iterator)
                opt = utils.get_optimizer(self.hparams, self.learning_rate)
                self.loss = model.loss
                self.params = self.Model.trainable_variables()

                gradients = opt.compute_gradients(
                    self.loss,
                    var_list=self.params,
                    colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

                clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
                    [grad for grad, _ in gradients], max_gradient_norm=MAX_GRADIENT_NORM)
                self.grad_norm = grad_norm

                self.update = opt.apply_gradients(zip(clipped_grads, self.params), self._global_step)

            self._summary = tf.summary.merge([
                tf.summary.scalar('train_loss', self.loss),
                tf.summary.scalar("learning_rate", self.learning_rate),
            ])

            self.train_model = model

        if eval or self.eval_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.eval_model = self.Model()
                self.eval_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batched_input_eval.iterator)
                self._eval_summary = tf.no_op()

        self.print_logs()

    def reset_train_iterator(self, sess):
        if self.train_mode:
            self._batched_input_train.reset_iterator(
                sess,
                skip=self.processed_inputs_count % self.data_size,
                #shuffle=self.epoch > 5
            )

    def print_logs(self):
        print("# Variables")
        for param in tf.global_variables():
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

        if self.train_mode:
            print("# Trainable variables")
            for param in self.params:
                print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

    def train(self, sess):
        try:
            self.processed_inputs_count, _, loss, _, summary, _, _= sess.run([
                self._processed_inputs_count,
                self.update,
                self.loss,
                self._global_step,
                self._summary,
                self.increment_inputs_count,
                self.train_model.get_extra_ops(),
            ])
        except tf.errors.OutOfRangeError:
            self.processed_inputs_count, _ = \
                sess.run([self._processed_inputs_count, self.increment_inputs_count])
            self.reset_train_iterator(sess)
            return self.train(sess)
        return loss, summary

    def eval(self, sess):
        input_filenames, target_labels, sample_ids, loss, summary, _ = sess.run([
            self.eval_model.input_filenames,
            self.eval_model.target_labels,
            self.eval_model.sample_id,
            self.eval_model.loss, self._eval_summary,
            self.eval_model.get_extra_ops()
        ])
        return input_filenames, target_labels, sample_ids, summary

    def eval_all(self, sess):
        lers = []
        self._batched_input_eval.reset_iterator(sess)
        while True:
            try:
                _, target_labels, decoded, _ = self.eval(sess)
                for i in range(len(target_labels)):
                    ler, _, _ = ops_utils.calculate_ler(
                        target_labels[i], decoded[i],
                        self._batched_input_eval.decode)
                    lers.append(ler)
            except tf.errors.OutOfRangeError:
                break

        return sum(lers) / len(lers)

    def _get_learning_rate_warmup(self, hparams):
        return self.learning_rate
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
        start_decay_epoch = 8
        decay_steps = 1
        decay_rate = 0.5

        return self.learning_rate if self.epoch < start_decay_epoch else \
                   tf.train.exponential_decay(
                       self.learning_rate,
                       (self.epoch - start_decay_epoch),
                       decay_steps, decay_rate, staircase=True)

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
        return self.processed_inputs_count / self.batch_size

    @property
    def data_size(self):
        return self._batched_input_train.size

    @property
    def epoch(self):
        return self.processed_inputs_count // self.data_size + 1

    @property
    def epoch_exact(self):
        return self.processed_inputs_count / self.data_size

    @property
    def epoch_progress(self):
        return (self.processed_inputs_count % self.data_size) // self.batch_size

    @property
    def step_size(self):
        return self.batch_size

