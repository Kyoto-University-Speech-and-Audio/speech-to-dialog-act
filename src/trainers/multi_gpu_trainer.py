import tensorflow as tf
from src.utils import model_utils, gpu_utils, utils, ops_utils
from .trainer import Trainer

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class MultiGPUTrainer(Trainer):
    def __init__(self, hparams, Model, BatchedInput, mode):
        super().__init__(hparams, Model, BatchedInput, mode)
        self.num_gpus = len(gpu_utils.get_available_gpus())
        self.increment_inputs_count = tf.assign(
            self._processed_inputs_count,
            self._processed_inputs_count + (self.hparams.batch_size * self.num_gpus))

    def build_model(self, eval=False):
        if self.train_mode:
            with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
                self.learning_rate = tf.constant(self.hparams.learning_rate)
                #self.learning_rate = self._get_learning_rate_warmup(self.hparams)
                #self.learning_rate = self._get_learning_rate_decay()

                opt = utils.get_optimizer(self.hparams, self.learning_rate)

                tower_grads = []
                losses = []
                controller = "/cpu:0"

                for i, id in enumerate(gpu_utils.get_available_gpus()):
                    name = 'tower_%d' % i
                    with tf.device(gpu_utils.assign_to_device(id, controller)), tf.name_scope(name):
                        model = self.Model()
                        model(
                            self.hparams,
                            tf.estimator.ModeKeys.TRAIN,
                            self._batched_input_train.iterator)
                        loss = model.loss
                        with tf.name_scope("compute_gradients"):
                            grad_and_vars = opt.compute_gradients(
                                loss,
                                colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)
                            vars = [var for _, var in grad_and_vars]
                            grads, _, _ = model_utils.gradient_clip([grad for grad, var in grad_and_vars], max_gradient_norm=MAX_GRADIENT_NORM)
                            tower_grads.append(zip(grads, vars))
                        losses.append(loss)
                    outer_scope.reuse_variables()
                self.train_model = model
                self.params = model.trainable_variables()

            with tf.name_scope("apply_gradients"), tf.device(controller):
                average_grads = []
                for grad_and_vars in zip(*tower_grads):
                    grads = [g for g, _ in grad_and_vars]
                    grad = tf.reduce_mean(grads, 0)
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)

                self.update = opt.apply_gradients(average_grads, self._global_step)
                self.loss = tf.reduce_mean(losses)

            self._summary = tf.summary.merge([
                tf.summary.scalar('train_loss', self.loss),
                tf.summary.scalar("learning_rate", self.learning_rate),
            ])

        if eval or self.eval_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                self.eval_model = self.Model()
                self.eval_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batched_input_eval.iterator)
                self._eval_summary = tf.no_op()

        self.print_logs()


    def reset_train_iterator(self, sess):
        self._batched_input_train.reset_iterator(
            sess,
            skip=self.processed_inputs_count % self.data_size,
            shuffle=self.epoch > 5)

    def train(self, sess):
        try:
            self.processed_inputs_count, _, loss, _, summary, _ = sess.run([
                self._processed_inputs_count,
                self.update,
                self.loss,
                self._global_step,
                self._summary,
                self.increment_inputs_count
            ])
        except tf.errors.OutOfRangeError:
            self.processed_inputs_count, _ = \
                sess.run([self._processed_inputs_count, self.increment_inputs_count])
            self.reset_train_iterator(sess)
            return self.train(sess)

        return loss, summary

    def eval(self, sess):
        input_filenames, target_labels, sample_ids, summary = sess.run([
            self.eval_model.input_filenames,
            self.eval_model.target_labels,
            self.eval_model.sample_id,
            self._eval_summary
        ])
        return input_filenames, target_labels, sample_ids, summary

    @property
    def global_step(self):
        return self.processed_inputs_count / self.hparams.batch_size

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
        return (self.processed_inputs_count % self.data_size) // (self.hparams.batch_size * self.num_gpus)

    @property
    def step_size(self):
        return self.hparams.batch_size * self.num_gpus
