import tensorflow as tf
from src.utils import model_utils, gpu_utils, utils, ops_utils
from .trainer import Trainer
import numpy as np

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
                self._train_models = []
                for i, id in enumerate(gpu_utils.get_available_gpus()):
                    name = 'tower_%d' % i
                    with tf.device(gpu_utils.assign_to_device(id, controller)), tf.name_scope(name):
                        model = self.Model()
                        model(
                            self.hparams,
                            tf.estimator.ModeKeys.TRAIN,
                            self._batched_input_train)
                        loss = model.loss
                        with tf.name_scope("compute_gradients"):
                            grad_and_vars = opt.compute_gradients(
                                loss,
                                var_list=model.trainable_variables(),
                                colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)
                            vars = [var for _, var in grad_and_vars]
                            grads, _, _ = model_utils.gradient_clip([grad for grad, var in grad_and_vars], max_gradient_norm=MAX_GRADIENT_NORM)
                            tower_grads.append(zip(grads, vars))
                        losses.append(loss)
                    outer_scope.reuse_variables()
                self._train_models.append(model)
                self._train_model = model
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

        # init dev model
        if self.hparams.dev_data is not None:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.dev_model = self.Model()
                self.hparams.batch_size = self.hparams.eval_batch_size
                self.dev_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batched_input_dev)

        if eval or self.eval_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                self.test_model = self.Model()
                self.hparams.batch_size = self.hparams.eval_batch_size
                self.test_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batched_input_test)
                self._eval_summary = tf.no_op()

        self.print_logs()

    def decay_batch_size(self, progress, sess):
        return False
        if progress > 0.5: batch_size = self.hparams.batch_size // 2
        elif progress > 0.8: batch_size = self.hparams.batch_size // 3
        else: batch_size = self.hparams.batch_size
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            tf.assign(self._batch_size, tf.constant(self.batch_size))
            self._batched_input_train.reset_iterator(
                sess,
                skip=self.processed_inputs_count % self.data_size,
            )
            return True
        return False

    def reset_train_iterator(self, sess):
        self._batched_input_train.reset_iterator(
            sess,
            skip=self.processed_inputs_count % self.data_size,
            shuffle=self.hparams.shuffle)

    def train(self, sess):
        model = self._train_models[0]
        try:
            ret = sess.run([
                    self._processed_inputs_count,
                    self.loss,
                    self._global_step,
                    self._summary,
                    self.increment_inputs_count
                ] + \
                ([self.update] if not self.hparams.simulated else []) + \
                model.get_ground_truth_label_placeholder() + \
                model.get_predicted_label_placeholder() + \
                model.get_extra_ops()
            )

            self.processed_inputs_count = ret[0]
            loss = ret[1]
            summary = ret[3]

            if self.hparams.output_result:
                target_ids = ret[-3]
                sample_ids = ret[-2]
                atts = ret[-1]
                with open(self.hparams.result_output_file, "a") as f:
                    for ids1, ids2, att in zip(target_ids, sample_ids,
                            atts):
                        _ids1 = [str(id) for id in ids1 if id < self.hparams.num_classes - 2]
                        _ids2 = [str(id) for id in ids2 if id < self.hparams.num_classes - 2]
                        fn = "%s/%d.npy" % (self.hparams.result_output_folder, self._eval_count)
                        f.write('\t'.join([
                            #filename.decode(),
                            ' '.join(_ids1),
                            ' '.join(_ids2),
                            #' '.join(self._batched_input_test.decode(ids2)),
                            fn
                        ]) + '\n')
                        att = att[:len(_ids2), :]
                        np.save(fn, att)
                        self._eval_count += 1

            #if self.hparams.verbose:
            #    print("\nprocessed_inputs_count:", self.processed_inputs_count)
            #    print("input_size:", len(inputs[0]))
            #    print("input_size:", len(inputs[1]))
            #    print("target[0]:", targets[0])
        except tf.errors.OutOfRangeError:
            self.processed_inputs_count, _ = \
                sess.run([self._processed_inputs_count, self.increment_inputs_count])
            self.reset_train_iterator(sess)
            return self.train(sess)

        return loss, summary

    def eval(self, sess, dev=False):
        model = self.dev_model if dev else self.test_model
        target_labels, sample_ids, summary = sess.run([
            model.get_ground_truth_label_placeholder(),
            model.get_predicted_label_placeholder(),
            self._eval_summary
        ])
        return target_labels, sample_ids, summary

    @property
    def global_step(self):
        return self.processed_inputs_count / self.hparams.batch_size

    @property
    def data_size(self):
        return self._batched_input_train.size or (self.hparams.train_size if self.train_mode else self.hparams.eval_size)

    @property
    def epoch(self):
        return self.processed_inputs_count // self.data_size + 1

    @property
    def epoch_exact(self):
        return self.processed_inputs_count / self.data_size + 1

    @property
    def epoch_progress(self):
        return (self.processed_inputs_count % self.data_size) // (self.batch_size * self.num_gpus)

    @property
    def step_size(self):
        return self.batch_size * self.num_gpus
