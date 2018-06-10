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

from tensorflow.python import debug as tf_debug

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--reset', type="bool", const=True, nargs="?", default=False)

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="aps")
    parser.add_argument('--model', type=str, default="ctc-attention")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")

    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

class ModelWrapper:
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        with self.graph.as_default():
            self.batched_input = BatchedInput(hparams, mode)
            self.batched_input.init_dataset()
            hparams.num_classes = self.batched_input.num_classes
            self.iterator = self.batched_input.iterator
            self.model = Model(
                hparams,
                mode=mode,
                iterator=self.iterator
            )

    def train(self, sess):
        return self.model.train(sess)

    def save(self, sess, global_step):
        self.model.saver.save(sess, os.path.join(self.hparams.out_dir, "csp.epoch%d.ckpt" % (global_step * self.hparams.batch_size // self.batched_input.size)))
        tf.logging.info('\n- Saved')

    def create_model(self, sess, name):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        global_step = self.model.global_step.eval(session=sess)
        return self.model, global_step

    def load_model(self, sess, name):
        latest_ckpt = tf.train.latest_checkpoint(self.hparams.out_dir)
        if latest_ckpt:
            self.model.saver.restore(sess, latest_ckpt)
            sess.run(tf.tables_initializer())
            global_step = self.model.global_step.eval(session=sess)
            return self.model, global_step
        else: return self.create_model(sess, name)

def train(Model, BatchedInput, hparams):
    train_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.TRAIN,
        BatchedInput, Model
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    train_sess = tf.Session(graph=train_model.graph, config=config)
    
    with train_model.graph.as_default():
        loaded_train_model, global_step = train_model.create_model(train_sess, "train") \
            if FLAGS.reset else train_model.load_model(train_sess, "train")
        data_size = train_model.batched_input.size
        skip = global_step * hparams.batch_size % data_size
        train_model.batched_input.reset_iterator(train_sess, skip=skip, shuffle=True)

    train_writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir, "log_train"), train_sess.graph)

    last_save_step = global_step
    last_eval_step = global_step
    epoch_batch_count = data_size // hparams.batch_size
    
    def get_pbar(initial=0):
        bar_format = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}{postfix}"
        return tqdm(total=epoch_batch_count,
                bar_format=bar_format, dynamic_ncols=True,
                initial=initial)

    pbar = get_pbar((global_step * hparams.batch_size % data_size) // hparams.batch_size)
    while True:
        train_cost = 0
        start = time.time()
        try:
            batch_cost, global_step = loaded_train_model.train(train_sess)
            # hparams.epoch_step += 1
            if global_step * hparams.batch_size % data_size < hparams.batch_size:
                pbar = get_pbar()
        except tf.errors.OutOfRangeError:
            # hparams.epoch_step = 0
            # print("Epoch completed")
            # pbar = get_pbar()
            train_model.batched_input.reset_iterator(train_sess, shuffle=True)
            continue

        train_cost += batch_cost * hparams.batch_size
        train_writer.add_summary(train_model.model.summary, global_step)

        train_cost /= hparams.batch_size

        pbar.update(1)
        pbar.set_postfix(cost="%.3f" % (train_cost))
        pbar.set_description('Epoch %i' % (global_step * hparams.batch_size // train_model.batched_input.size + 1))

        if global_step - last_save_step >= 300:
            train_model.save(train_sess, global_step)
            last_save_step = global_step

        if global_step - last_eval_step >= 3000:
            hparams.beam_width = 0
            eval(hparams, output=False)
            last_eval_step = global_step

def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)

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
