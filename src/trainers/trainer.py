import tensorflow as tf
import numpy as np
from utils import model_utils, utils, ops_utils

WARMUP_STEPS = 0
WARMUP_SCHEME = 't2t'
DECAY_SCHEME = ''


class Trainer(object):
    def __init__(self, hparams, Model, BatchedInput, mode, graph=None):
        self.hparams = hparams
        self.mode = mode
        self.graph = graph
        self.train_mode = self.mode == tf.estimator.ModeKeys.TRAIN
        self.eval_mode = self.mode == tf.estimator.ModeKeys.EVAL
        self.infer_mode = self.mode == tf.estimator.ModeKeys.PREDICT

        self.Model = Model
        self.BatchedInput = BatchedInput
        self.batch_size = self.hparams.eval_batch_size if self.eval_mode else self.hparams.batch_size
        self._batch_size = tf.Variable(
            self.batch_size,
            trainable=False,
            name="batch_size")
        self.eval_batch_size = self.hparams.eval_batch_size
        self._eval_batch_size = tf.Variable(
            self.eval_batch_size,
            trainable=False, name="eval_batch_size")

        batched_input_test = None
        batched_input_dev = None
        batched_input_infer = None

        # init train set and dev set
        if self.train_mode:
            if hparams.dev_data is not None:
                batched_input_dev = BatchedInput(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batch_size,
                    dev=True)
                batched_input_dev.init_dataset()

            batched_input_train = BatchedInput(
                self.hparams,
                tf.estimator.ModeKeys.TRAIN,
                self._eval_batch_size)
            batched_input_train.init_dataset()
        else:
            batched_input_train = None

        # init test set (for train or eval)
        if self.train_mode or self.eval_mode:
            if hparams.test_data is None:
                raise Exception("No testing input")

            batched_input_test = BatchedInput(
                self.hparams,
                tf.estimator.ModeKeys.EVAL,
                self.eval_batch_size,
                dev=False
            )
            batched_input_test.init_dataset()
        else:  # infer mode
            batched_input_infer = BatchedInput(
                self.hparams,
                tf.estimator.ModeKeys.PREDICT,
                1,
            )
            batched_input_infer.init_dataset()

        self.hparams.vocab_size = \
            (batched_input_train or batched_input_test or batched_input_infer).vocab_size

        self._processed_inputs_count = tf.Variable(0, trainable=False)
        self.processed_inputs_count = 0
        self.increment_inputs_count = tf.assign(
            self._processed_inputs_count,
            self._processed_inputs_count + self.batch_size)

        self._global_step = tf.Variable(0, trainable=False)

        self._batched_input_train = batched_input_train
        self._batched_input_test = batched_input_test
        self._batched_input_dev = batched_input_dev
        self._batched_input_infer = batched_input_infer

        self._eval_count = 0

    def init(self, sess):
        self.processed_inputs_count = sess.run(self._processed_inputs_count)
        self.reset_train_iterator(sess)
        if self._batched_input_test is not None:
            self._batched_input_test.reset_iterator(sess)
        if self._batched_input_infer is not None:
            self._batched_input_infer.reset_iterator(sess)
        if self._batched_input_dev is not None:
            self._batched_input_dev.reset_iterator(sess)

    def build_model(self, eval=False):
        if self.train_mode:
            with tf.variable_scope(tf.get_variable_scope()):
                self.learning_rate = tf.constant(self.hparams.learning_rate)
                # self.learning_rate = self._get_learning_rate_warmup(self.hparams)
                self.learning_rate = self._get_learning_rate_decay()

                # build model
                model = self.Model()
                model(
                    self.hparams,
                    tf.estimator.ModeKeys.TRAIN,
                    self._batched_input_train)
                opt = utils.get_optimizer(self.hparams, self.learning_rate)
                self.loss = model.loss
                self.params = model.trainable_variables()

                # compute gradient & update vảiables
                gradients = opt.compute_gradients(
                    self.loss,
                    var_list=self.params,
                    colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

                clipped_grads, grad_norm_summary, grad_norm = model_utils.gradient_clip(
                    [grad for grad, _ in gradients], max_gradient_norm=self.hparams.max_gradient_norm)
                self.grad_norm = grad_norm

                self.update = opt.apply_gradients(zip(clipped_grads, self.params), self._global_step)

            self._summary = tf.summary.merge([
                tf.summary.scalar('train_loss', self.loss),
                tf.summary.scalar("learning_rate", self.learning_rate),
            ])

            self._train_model = model

            # init dev model
            if self.hparams.dev_data is not None:
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    self.dev_model = self.Model()
                    self.dev_model(
                        self.hparams,
                        tf.estimator.ModeKeys.EVAL,
                        self._batched_input_dev)

        # init test model
        if eval > 0 or self.eval_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.test_model = self.Model()
                self.test_model(
                    self.hparams,
                    tf.estimator.ModeKeys.EVAL,
                    self._batched_input_test)
                self._eval_summary = tf.no_op()

        if self.infer_mode:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                self.infer_model = self.Model()
                self.infer_model(
                    self.hparams,
                    tf.estimator.ModeKeys.PREDICT,
                    self._batched_input_infer)

        if self.hparams.verbose:
            self.print_logs()

    def reset_train_iterator(self, sess):
        if self.train_mode:
            self._batched_input_train.reset_iterator(
                sess,
                skip=self.processed_inputs_count % self.data_size,
                # shuffle=self.epoch > 5
            )
            self._train_model._assign_input()

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
        model = self._train_model
        try:
            ret = model.train(sess, [
                                        self._processed_inputs_count,
                                        self.loss,
                                        self._global_step,
                                        self._summary,
                                        self.increment_inputs_count,
                                    ] + \
                                    [self.update] if not self.hparams.is_simulated_train else [] + \
                                                                                              model.get_ground_truth_label_placeholder() + \
                                                                                              model.get_predicted_label_placeholder() + \
                                                                                              model.get_extra_ops()
                              )
            self.processed_inputs_count = ret[0]

            if self.hparams.result_output_file:
                target_ids = ret[-4]
                sample_ids = ret[-3]
                encoder_outputs = ret[-2]
                encoder_state = ret[-1]
                # atts = ret[-1]
                with open(self.hparams.result_output_file, "a") as f:
                    for ids1, ids2, eo, es, filename in zip(target_ids, sample_ids,
                                                            eo, es, ret[2]):
                        _ids1 = [str(id) for id in ids1 if id < self.hparams.vocab_size - 2]
                        _ids2 = [str(id) for id in ids2 if id < self.hparams.vocab_size - 2]
                        fn_eo = "%s/eo_%d.npy" % (self.hparams.result_output_folder, self._eval_count)
                        fn_es = "%s/es_%d.npy" % (self.hparams.result_output_folder, self._eval_count)
                        f.write('\t'.join([
                            filename.decode(),
                            ' '.join(_ids1),
                            ' '.join(_ids2),
                            # ' '.join(self._batched_input_test.decode(ids2)),
                            fn_eo,
                            fn_es
                        ]) + '\n')
                        att = att[:len(_ids2), :]
                        np.save(fn, att)
                        self._eval_count += 1

        except tf.errors.OutOfRangeError:
            self.processed_inputs_count, _ = \
                sess.run([self._processed_inputs_count, self.increment_inputs_count])
            self.reset_train_iterator(sess)
            return self.train(sess)
        return loss, summary

    def eval(self, sess, dev=False):
        model = self.dev_model if dev else self.test_model
        # number of evaluated accuracies (more than 1 for multi-task learning
        num_acc = len(model.get_ground_truth_label_placeholder())
        ret = sess.run(
            model.get_ground_truth_label_placeholder() + \
            model.get_predicted_label_placeholder() + \
            model.get_ground_truth_label_len_placeholder() + \
            model.get_predicted_label_len_placeholder() + \
            # [model.input_filenames] + \
            [
                model.loss, self._eval_summary,
            ] + model.get_extra_ops()
        )

        if self.hparams.output_result:
            model.output_result(
                ret[:num_acc],
                ret[num_acc:2 * num_acc],
                ret[2 * num_acc:3 * num_acc],
                ret[3 * num_acc:4 * num_acc],
                ret[-len(model.get_extra_ops()):],
                self._eval_count
            )
            self._eval_count += len(ret[0])

        return None, ret[:num_acc], ret[num_acc:2 * num_acc], \
               ret[2 * num_acc:3 * num_acc], ret[3 * num_acc:4 * num_acc]

    def infer(self, sess=None):
        model = self.infer_model
        sess = sess or self.sess
        self._batched_input_infer.reset_iterator(sess)

        ret = sess.run(
            model.get_predicted_label_placeholder() + \
            model.get_predicted_label_len_placeholder()
        )

        sample_ids = ret[0]
        decode_fns = self.infer_model.get_decode_fns()
        return [''.join(decode_fns[0](ids)) for ids in sample_ids]

    def eval_all(self, sess, dev=False):
        """
        Iterate through dataset and return final accuracy

        Returns
            dictionary of key: id, value: accuracy
        """
        lers = {}
        decode_fns = self.test_model.get_decode_fns()
        metrics = self.hparams.metrics.split(',')

        batched_input = self._batched_input_dev if dev else self._batched_input_test
        if batched_input is None: return None
        batched_input.reset_iterator(sess)
        while True:
            try:
                _, ground_truth_labels, predicted_labels, ground_truth_len, predicted_len = self.eval(sess, dev)
                for acc_id, (gt_labels, p_labels, gt_len, p_len) in \
                        enumerate(zip(ground_truth_labels, predicted_labels, ground_truth_len, predicted_len)):
                    if acc_id not in lers: lers[acc_id] = []
                    for i in range(len(gt_labels)):
                        if acc_id == 1 and self.hparams.model == "da_attention_seg":
                            ler, str_original, str_decoded = ops_utils.joint_evaluate(
                                self.hparams,
                                ground_truth_labels[0][i], predicted_labels[0][i],
                                ground_truth_labels[1][i], predicted_labels[1][i],
                                decode_fns[acc_id],
                            )
                        else:
                            ler, _, _ = ops_utils.evaluate(
                                gt_labels[i],  # [:gt_len[i]],
                                p_labels[i],  # [:p_len[i]],
                                decode_fns[acc_id],
                                metrics[acc_id], acc_id)
                        if ler is not None: lers[acc_id].append(ler)
            except tf.errors.OutOfRangeError:
                break

        return {acc_id: sum(lers[acc_id]) / len(lers[acc_id]) for acc_id in lers}

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
        return self.learning_rate if self.epoch < self.hparams.learning_rate_start_decay_epoch else \
            tf.train.exponential_decay(
                self.learning_rate,
                (self.epoch - self.hparams.learning_rate_start_decay_epoch),
                self.hparams.learning_rate_decay_steps,
                self.hprams.learning_rate_decay_rate, staircase=True)

    def _build_graph(self):
        pass

    def load(self, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        # if flags.load_ignore_scope:
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
        return (self._batched_input_train.size if self.train_mode else self._batched_input_test.size) \
               or (self.hparams.train_size if self.train_mode else self.hparams.eval_size)

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
