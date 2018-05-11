import argparse
import os
import tensorflow as tf
import random
import numpy as np
import importlib
import sys
import time

sys.path.insert(0, os.path.abspath('.'))

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--dataset', type=str, default="vctk")
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_encoder_layers", type=int, default=2,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=2,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default="tmp",
                        help="Store log/model files.")

def create_hparams(flags):
    return tf.contrib.training.HParams(
        num_units=flags.num_units,
        num_encoder_layers=flags.num_encoder_layers,
        num_decoder_layers=flags.num_decoder_layers,
        batch_size=flags.batch_size,
        summaries_dir=flags.summaries_dir,
        out_dir=flags.out_dir,
        num_train_steps=flags.num_train_steps,

        epoch_step=0,
    )

class ModelWrapper:
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        with self.graph.as_default():
            self.batched_input = BatchedInput(mode, hparams.batch_size)
            self.iterator = self.batched_input.iterator
            self.model = Model(
                hparams,
                mode=mode,
                iterator=self.iterator
            )

    def train(self, sess):
        return self.model.train(sess)

    def save(self, sess, global_step):
        tf.logging.info('Saving to "%s-%d"', self.hparams.out_dir, global_step)
        self.model.saver.save(sess, os.path.join(self.hparams.out_dir, "csp.ckpt"))

    def create_or_load_model(self, sess, name):
        latest_ckpt = tf.train.latest_checkpoint(self.hparams.out_dir)
        if latest_ckpt:
            self.model.saver.restore(sess, latest_ckpt)
            sess.run(tf.tables_initializer())
            return self.model, 0
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            return self.model, 0

def train(Model, BatchedInput, hparams):
    hparams.num_classes = BatchedInput.num_classes
    train_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.TRAIN,
        BatchedInput, Model
    )
    eval_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.EVAL,
        BatchedInput, Model
    )

    train_sess = tf.Session(graph=train_model.graph)
    eval_sess = tf.Session(graph=eval_model.graph)

    with train_model.graph.as_default():
        train_model.batched_input.reset_iterator(train_sess)

    with train_model.graph.as_default():
        loaded_train_model, global_step = train_model.create_or_load_model(train_sess, "train")

    global_step = 0

    train_writer = tf.summary.FileWriter(hparams.summaries_dir + '/train', train_sess.graph)
    validation_writer = tf.summary.FileWriter(hparams.summaries_dir + '/validation')

    last_eval_step = global_step
    while global_step < hparams.num_train_steps:
        train_cost = train_ler = 0
        start = time.time()

        try:
            batch_cost, ler = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            hparams.epoch_step = 0
            global_step += 1
            train_model.batched_input.reset_iterator(train_sess)
            continue

        train_cost += batch_cost * hparams.batch_size
        train_ler += ler * hparams.batch_size
        train_writer.add_summary(train_model.model.summary, global_step * hparams.batch_size + hparams.epoch_step)

        train_cost /= hparams.batch_size
        train_ler /= hparams.batch_size

        # val_cost, val_ler = sess.run([self.cost, self.ler])
        val_cost, val_ler = 0, 0

        log = "Epoch {}/{}, Batch {}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        tf.logging.info(log.format(global_step + 1, hparams.num_train_steps,
                                   hparams.epoch_step,
                                   train_cost, train_ler,
                                   time.time() - start))

        if global_step - last_eval_step >= 1:
            train_model.save(train_sess, global_step)

            with eval_model.graph.as_default():
                loaded_eval_model, _ = eval_model.create_or_load_model(eval_sess, "eval")

            with eval_model.graph.as_default():
                loaded_eval_model, _ = eval_model.create_or_load_model(
                    eval_sess, "eval"
                )

                eval_model.batched_input.reset_iterator(eval_sess)
                target_labels, test_cost, test_ler, decoded = loaded_eval_model.decode(eval_sess)

                log = "Epoch {}/{}:, test_cost = {:.3f}, test_ler = {:.3f}"
                tf.logging.info(log.format(global_step + 1, hparams.num_train_steps,
                                           test_cost, test_ler))

            for i in range(min(3, len(target_labels))):
                str_original = BatchedInput.decode(target_labels[i])
                str_decoded = BatchedInput.decode(decoded[i])

                print('Original: %s' % str_original)
                print('Decoded:  %s' % str_decoded)

            last_eval_step = global_step

def main(unused_argv):
    hparams = create_hparams(FLAGS)

    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if FLAGS.dataset == 'vivos':
        from input_data.vivos import BatchedInput
    elif FLAGS.dataset == 'vctk':
        from input_data.vctk import BatchedInput

    if FLAGS.model == 'ctc':
        from models.ctc import CTCModel as Model
    elif FLAGS.model == 'attention':
        from models.attention import AttentionModel as Model

    train(Model, BatchedInput, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)