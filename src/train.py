import argparse
import os
import tensorflow as tf
import random
import numpy as np
import sys
from . import configs
from .utils import utils
from tqdm import tqdm
from .trainers.multi_gpu_trainer import MultiGPUTrainer
from .trainers.trainer import Trainer
from tensorflow.python import debug as tf_debug

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--reset', type="bool", const=True, nargs="?", default=False,
                        help="No saved model loaded")
    parser.add_argument('--debug', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--eval', type=int, const=True, nargs="?", default=0,
                        help="Frequently check and log evaluation result")
    parser.add_argument('--gpus', type="bool", const=True, nargs="?", default=False,
                        help="Use MultiGPUTrainer")
    parser.add_argument('--transfer', type="bool", const=True, nargs="?", default=False,
                        help="If model needs custom load.")

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


def load_model(sess, Model, hparams):
    sess.run(tf.global_variables_initializer())
    if FLAGS.reset: return
    if FLAGS.load:
        ckpt = os.path.join(hparams.out_dir, "csp.%s.ckpt" % FLAGS.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)
    if ckpt:
        if FLAGS.transfer:
            saver_variables = tf.global_variables()
            var_list = {var.op.name: var for var in saver_variables}
            del var_list["apply_gradients/beta1_power"]
            del var_list["apply_gradients/beta2_power"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Adam_1"]
            del var_list["stack_bidirectional_rnn/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Adam_1"]
            saver = tf.train.Saver(var_list)
            saver.restore(sess, ckpt)

            #Model.load(sess, ckpt, FLAGS)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)


def argval(name):
    return utils.argval(name, FLAGS)


def train(Model, BatchedInput, hparams):
    hparams.beam_width = 0
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.TRAIN
    with graph.as_default():
        new_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder/correction')
        init_new_vars = tf.initialize_variables(new_vars)

        if FLAGS.gpus: trainer = MultiGPUTrainer(hparams, Model, BatchedInput, mode)
        else: trainer = Trainer(hparams, Model, BatchedInput, mode)
        trainer.build_model(eval=argval("eval"))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(graph=graph, config=config)
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        load_model(sess, Model, hparams)

        trainer.init(sess)

        writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir, "log"), sess.graph)

        last_save_step = trainer.global_step
        last_eval_pos = trainer.global_step - FLAGS.eval
        epoch_batch_count = trainer.data_size // trainer.step_size

        def reset_pbar():
            pbar = tqdm(total=epoch_batch_count,
                    ncols=150,
                    unit="step",
                    initial=trainer.epoch_progress)
            pbar.set_description('Epoch %i' % trainer.epoch)
            return pbar

        pbar = reset_pbar()
        last_epoch = trainer.epoch
        ler = 1
        min_ler = 1

        while True:
            loss, summary = trainer.train(sess)

            if trainer.epoch > last_epoch:
                pbar = reset_pbar()
                last_epoch = trainer.epoch

            train_cost = loss
            writer.add_summary(summary, trainer.processed_inputs_count)
            pbar.update(1)
            pbar.set_postfix(cost="%.3f" % (train_cost), min_ler="%2.1f" % (min_ler * 100), last_ler="%2.1f" % (ler * 100))

            if trainer.global_step - last_save_step >= FLAGS.save_steps:
                path = os.path.join(
                    hparams.out_dir,
                    "csp.epoch%d.ckpt" % (trainer.epoch))
                saver = tf.train.Saver()
                saver.save(sess, path)
                last_save_step = trainer.global_step

            if argval("eval"):
                if trainer.global_step - last_eval_pos >= FLAGS.eval:
                    pbar.set_postfix_str("Evaluating...")
                    ler = trainer.eval_all(sess)
                    if ler < min_ler: min_ler = ler
                    writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(simple_value=ler, tag="label_error_rate")]),
                        trainer.processed_inputs_count)
                    last_eval_pos = trainer.global_step


def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = os.path.join(configs.HCOPY_CONFIG_PATH)

    utils.clear_log()

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
