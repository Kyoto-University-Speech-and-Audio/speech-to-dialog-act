import argparse
import os
import tensorflow as tf
import random
import numpy as np
import sys
import shutil
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
    parser.add_argument('--verbose', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--reset', type="bool", const=True, nargs="?", default=False,
                        help="No saved model loaded")
    parser.add_argument('--debug', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--shuffle', type="bool", const=True, nargs="?", default=False,
                        help="Shuffle dataset")
    parser.add_argument('--eval', type=int, default=1000,
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
            """
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
            """
            Model.load(sess, ckpt, FLAGS)
        else:
            saver_variables = tf.global_variables()
            var_list = {var.op.name: var for var in saver_variables}
            for var in Model.ignore_save_variables():
                print(var_list[var])
                if var in var_list:
                    del var_list[var]
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, ckpt)

def save(hparams, sess, name=None):
    path = os.path.join(
        hparams.out_dir,
        "csp.%s.ckpt" % (name))
    saver = tf.train.Saver()
    saver.save(sess, path)
    if hparams.verbose: print("Saved as csp.%s.ckpt" % (name))

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
        trainer.build_model(eval=argval("eval") != 0)

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

        # tensorboard log
        if FLAGS.reset:
            if os.path.exists(hparams.summaries_dir):
                shutil.rmtree(hparams.summaries_dir)
        writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir), sess.graph)

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
        dev_lers = {}
        min_dev_lers = {}
        test_lers = {}
        min_test_lers = {}
        min_dev_test_lers = {}

        while True:
            utils.update_hparams(FLAGS, hparams) # renew hparams so paramters can be changed during training
            
            loss, summary = trainer.train(sess)

            if trainer.epoch > last_epoch:
                pbar = reset_pbar()
                last_epoch = trainer.epoch

            writer.add_summary(summary, trainer.processed_inputs_count)
            pbar.update(1)

            if trainer.global_step - last_save_step >= FLAGS.save_steps:
                save(hparams, sess, "epoch%d" % trainer.epoch)
                last_save_step = trainer.global_step
            
            if trainer.epoch > hparams.max_epoch_num: break

            if argval("eval") > 0:
                if trainer.global_step - last_eval_pos >= FLAGS.eval:
                    pbar.set_postfix_str("Evaluating (dev)...")
                    dev_lers = trainer.eval_all(sess, dev=True)
                    pbar.set_postfix_str("Evaluating (test)...")
                    test_lers = trainer.eval_all(sess, dev=False)
                    
                    for acc_id in test_lers:
                        if dev_lers is None:
                            if acc_id not in min_test_lers or min_test_lers[acc_id] > test_lers[acc_id]:
                                min_test_lers[acc_id] = test_lers[acc_id]
                                save(hparams, sess, "best_%d" % acc_id)
                        else:
                            if acc_id not in min_test_lers or min_test_lers[acc_id] > test_lers[acc_id]:
                                min_test_lers[acc_id] = test_lers[acc_id]

                            if acc_id not in min_dev_lers or (min_dev_lers[acc_id] > dev_lers[acc_id]):
                                min_dev_lers[acc_id] = dev_lers[acc_id]
                                min_dev_test_lers[acc_id] = test_lers[acc_id]
                                save(hparams, sess, "best_%d" % acc_id)
                            
                            tqdm.write("dev: %2.1f, test: %2.1f, acc: %2.1f" %
                                    (dev_lers[acc_id], test_lers[acc_id], min_test_lers[acc_id]))
                    
                        for (err_id, lers) in [("dev", dev_lers), ("test", test_lers), ("min_test", min_test_lers)]:
                            if lers is not None:
                                writer.add_summary(
                                    tf.Summary(value=[tf.Summary.Value(simple_value=lers[acc_id],
                                        tag="%s_error_rate_%d" % (err_id, acc_id))]),
                                    trainer.processed_inputs_count)

                    last_eval_pos = trainer.global_step
            
            # update postfix
            pbar_pf = {}
            for acc_id in test_lers:
                pbar_pf["min_dev" + str(acc_id)] = "%2.1f" % (min_dev_test_lers[acc_id] * 100)
                pbar_pf["min_test" + str(acc_id)] = "%2.1f" % (min_test_lers[acc_id] * 100)
                pbar_pf["test" + str(acc_id)] = "%2.1f" % (test_lers[acc_id] * 100)
                pbar_pf["dev" + str(acc_id)] = "%2.1f" % (dev_lers[acc_id] * 100)
            pbar_pf['cost'] = "%.3f" % (loss)
            pbar.set_postfix(pbar_pf)


def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = os.path.join(configs.HCOPY_CONFIG_PATH)

    if not os.path.exists(hparams.summaries_dir): os.mkdir(hparams.summaries_dir)
    if not os.path.exists(hparams.out_dir): os.mkdir(hparams.out_dir)
    utils.clear_log(hparams)

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
