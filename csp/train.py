import argparse
import os
import tensorflow as tf
import random
import numpy as np
import importlib
import sys
import time
from .utils import utils
from tqdm import tqdm
from .eval import eval
from .models.trainers.multi_gpu_trainer import MultiGPUTrainer

from tensorflow.python import debug as tf_debug

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--reset', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--debug', type="bool", const=True, nargs="?", default=False)

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="aps")
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--model', type=str, default="ctc-attention")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")

    parser.add_argument('--load_ignore_scope', type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

def load_model(sess, hparams):
    sess.run(tf.global_variables_initializer())
    if FLAGS.reset: return
    if FLAGS.load:
        ckpt = os.path.join(hparams.out_dir, "csp.%s.ckpt" % FLAGS.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)

    if ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
    else:
        pass

def train(Model, BatchedInput, hparams):
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.TRAIN
    with graph.as_default():
        batched_input = BatchedInput(hparams, mode)
        batched_input.init_dataset()
        hparams.num_classes = batched_input.num_classes

        model_fn = lambda: Model(
            hparams, mode,
            batched_input.iterator
        )

        new_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder/correction')
        init_new_vars = tf.initialize_variables(new_vars)

        trainer = MultiGPUTrainer(hparams, mode)
        trainer(model_fn, batched_input)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(graph=graph, config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        load_model(sess, hparams)

        trainer.processed_inputs_count = sess.run(trainer._processed_inputs_count)

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        data_size = batched_input.size
        skip = trainer.processed_inputs_count % data_size
        batched_input.reset_iterator(
            sess, skip=skip,
            shuffle=trainer.epoch > 5)

        train_writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir, "log_train"), sess.graph)

        last_save_step = trainer.global_step
        last_eval_step = trainer.global_step - 3000
        epoch_batch_count = data_size // hparams.batch_size // trainer.num_gpus

        def get_pbar(initial=0):
            bar_format = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}"
            return tqdm(total=epoch_batch_count,
                    bar_format=bar_format, ncols=100,
                    initial=initial)

        pbar = get_pbar(trainer.epoch_progress)
        pbar.set_description('Epoch %i' % trainer.epoch)
        while True:
            train_cost = 0
            try:
                loss, summary = trainer.train(sess)
                # hparams.epoch_step += 1
                if trainer.epoch_progress == 0:
                    pbar = get_pbar()
            except tf.errors.OutOfRangeError:
                # hparams.epoch_step = 0
                # print("Epoch completed")
                # pbar = get_pbar()
                batched_input.reset_iterator(sess, shuffle=trainer.epoch > 5)
                continue

            train_cost += loss * hparams.batch_size
            train_writer.add_summary(summary, trainer.epoch_exact)

            train_cost /= hparams.batch_size

            pbar.update(1)
            pbar.set_postfix(cost="%.3f" % (train_cost))

            if trainer.global_step - last_save_step >= FLAGS.save_steps:
                path = os.path.join(
                    hparams.out_dir,
                    "csp.epoch%d.ckpt" % (trainer.epoch))
                saver = tf.train.Saver()
                saver.save(sess, path)
                print(trainer.processed_inputs_count)
                tf.logging.info('\n- Saved to ' + path)
                last_save_step = trainer.global_step

            if trainer.global_step - last_eval_step >= 3000:
                hparams.beam_width = 0
                #eval(hparams, FLAGS)
                last_eval_step = trainer.global_step

def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = "/n/sd7/trung/bin/htk/HTKTools/HCopy"
    # hparams.hcopy_path = os.path.join("bin", "htk", "bin.win32", "HCopy.exe")
    hparams.hcopy_config = os.path.join("/n/sd7/trung/config.lmfb.40ch")
    # hparams.hcopy_config = os.path.join("data", "config.lmfb.40ch")

    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)

    train(Model, BatchedInput, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
