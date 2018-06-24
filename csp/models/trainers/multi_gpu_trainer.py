import tensorflow as tf
from ...utils import model_utils, gpu_utils, utils
from .trainer import Trainer

MAX_GRADIENT_NORM = 5.0
WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''

class MultiGPUTrainer(Trainer):
    def __call__(self, model_fn, batched_input, **kwargs):
        self._processed_inputs_count = tf.Variable(0, trainable=False)
        self.processed_inputs_count = 0
        self.num_gpus = len(gpu_utils.get_available_gpus())
        self.increment_inputs_count = tf.assign(self._processed_inputs_count,
                      self._processed_inputs_count + (self.hparams.batch_size * self.num_gpus))

        self._global_step = tf.Variable(0, trainable=False)
        self._batched_input = batched_input
        self.iterator = batched_input.iterator

        self.learning_rate = tf.constant(self.hparams.learning_rate)
        self.learning_rate = self._get_learning_rate_warmup(self.hparams)
        self.learning_rate = self._get_learning_rate_decay()

        opt = utils.get_optimizer(self.hparams, self.learning_rate)

        tower_grads = []
        losses = []
        controller = "/cpu:0"

        self.params = self._get_trainable_params(**kwargs)

        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for i, id in enumerate(gpu_utils.get_available_gpus()):
                name = 'tower_%d' % i
                with tf.device(gpu_utils.assign_to_device(id, controller)), tf.name_scope(name):
                    model = model_fn()
                    loss = model.loss
                    self.params = self._get_trainable_params(**kwargs)
                    with tf.name_scope("compute_gradients"):
                        grad_and_vars = opt.compute_gradients(
                            loss,
                            colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

                        grads, _, _ = model_utils.gradient_clip(grad_and_vars, max_gradient_norm=MAX_GRADIENT_NORM)
                        tower_grads.append(grad_and_vars)
                    losses.append(loss)
                outer_scope.reuse_variables()
            self.model = model

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

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train_loss', self.loss),
            tf.summary.scalar("learning_rate", self.learning_rate),
        ])

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
        self.processed_inputs_count, _, loss, _, summary, _ = sess.run([
            self._processed_inputs_count,
            self.update,
            self.loss,
            self._global_step,
            self.train_summary,
            self.increment_inputs_count
        ])
        return loss, summary

    def eval(self, sess):
        input_filenames, target_labels, sample_ids, loss, summary = sess.run([
            self.input_filenames,
            self.target_labels,
            self.model.sample_id,
            self.loss, tf.no_op()
        ])
        return input_filenames, target_labels, sample_ids, summary

    @property
    def global_step(self):
        return self.processed_inputs_count / self.hparams.batch_size

    @property
    def data_size(self):
        return self._batched_input.size

    @property
    def epoch(self):
        return self.processed_inputs_count // self.data_size + 1

    @property
    def epoch_exact(self):
        return self.processed_inputs_count / self.data_size

    @property
    def epoch_progress(self):
        return (self.processed_inputs_count % self.data_size) // (self.hparams.batch_size * self.num_gpus)