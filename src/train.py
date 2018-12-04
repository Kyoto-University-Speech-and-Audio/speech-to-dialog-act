import argparse
import os
import tensorflow as tf
import random
import numpy as np
import sys
import shutil
import json
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
    parser.add_argument('--eval', type=int, default=0,
                        help="Frequently check and log evaluation result")
    parser.add_argument('--eval_from', type=int, default=0,
                        help="No. of epoch before eval")
    parser.add_argument('--transfer', type="bool", const=True, nargs="?", default=False,
                        help="If model needs custom load.")

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="aps")
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--output', type="bool", const=True, nargs="?", default=False)
    parser.add_argument('--simulated', type="bool", const=True, nargs="?", default=False)
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
            Model.load(sess, ckpt, FLAGS)
        else:
            saver_variables = tf.global_variables()
            var_list = {var.op.name: var for var in saver_variables}
            for var in Model.ignore_save_variables() + ['batch_size',
                                                        'eval_batch_size', 'Variable_1',
                                                        'apply_gradients/beta1_power', 'apply_gradients/beta2_power']:
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
        trainer = MultiGPUTrainer(hparams, Model, BatchedInput, mode)
        trainer.build_model(eval=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(graph=graph, config=config)
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        load_model(sess, Model, hparams)

        if argval("simulated"):
            # not real training, only to export values
            utils.prepare_output_path(hparams)
            sess.run(tf.assign(trainer._global_step, 0))
            sess.run(tf.assign(trainer._processed_inputs_count, 0))

        trainer.init(sess)

        # tensorboard log
        if FLAGS.reset:
            if os.path.exists(hparams.summaries_dir):
                shutil.rmtree(hparams.summaries_dir)
        writer = tf.summary.FileWriter(os.path.join(hparams.summaries_dir), sess.graph)

        last_save_step = trainer.global_step
        last_eval_pos = trainer.global_step - FLAGS.eval

        def reset_pbar():
            epoch_batch_count = trainer.data_size // trainer.step_size
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

        trainer.reset_train_iterator(sess)

        while True:
            # utils.update_hparams(FLAGS, hparams) # renew hparams so paramters can be changed during training

            # eval if needed
            if argval("eval") > 0 and argval("eval_from") < trainer.epoch_exact \
                or argval("eval") == 0 and trainer.epoch > last_epoch:
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

                        for (err_id, lers) in [("dev", dev_lers), ("test", test_lers), ("min_test", min_dev_test_lers)]:
                            if lers is not None and len(lers) > 0:
                                writer.add_summary(
                                    tf.Summary(value=[tf.Summary.Value(simple_value=lers[acc_id],
                                                                       tag="%s_error_rate_%d" % (err_id, acc_id))]),
                                    trainer.processed_inputs_count)

                    last_eval_pos = trainer.global_step

            loss, summary = trainer.train(sess)

            # return

            if trainer.epoch > last_epoch:  # reset epoch
                pbar = reset_pbar()
                last_epoch = trainer.epoch

            writer.add_summary(summary, trainer.processed_inputs_count)
            pbar.update(1)

            if not argval("simulated") and trainer.global_step - last_save_step >= FLAGS.save_steps:
                save(hparams, sess, "epoch%d" % trainer.epoch)
                last_save_step = trainer.global_step

            if trainer.epoch > hparams.max_epoch_num: break

            # reduce batch size with long input
            if hparams.batch_size_decay:
                if trainer.decay_batch_size(trainer.epoch_exact -
                                            trainer.epoch, sess):
                    pbar = reset_pbar()

            # update postfix
            pbar_pf = {}
            for acc_id in test_lers:
                if dev_lers is not None: pbar_pf["min_dev" + str(acc_id)] = "%2.2f" % (min_dev_test_lers[acc_id] * 100)
                pbar_pf["min_test" + str(acc_id)] = "%2.2f" % (min_test_lers[acc_id] * 100)
                pbar_pf["test" + str(acc_id)] = "%2.2f" % (test_lers[acc_id] * 100)
                if dev_lers is not None: pbar_pf["dev" + str(acc_id)] = "%2.2f" % (dev_lers[acc_id] * 100)
            pbar_pf['cost'] = "%.3f" % (loss)
            pbar.set_postfix(pbar_pf)


def main(unused_argv):
    if FLAGS.config is None: 
        raise Exception("Config file must be provided")
    
    json_file = open('model_configs/%s.json' % FLAGS.config).read()
    json_dict = json.loads(json_file)
    BatchedInput = utils.get_batched_input_class(json_dict.get("input", "default"))
    Model = utils.get_model_class(json_dict.get("model"))
    
    hparams = utils.create_hparams(FLAGS, Model)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = os.path.join(configs.HCOPY_CONFIG_PATH)

    # create directories for logs and saved parameters
    if not os.path.exists(hparams.summaries_dir): os.mkdir(hparams.summaries_dir)
    if not os.path.exists(hparams.out_dir): os.mkdir(hparams.out_dir)
    utils.clear_log(hparams)

    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    train(Model, BatchedInput, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
