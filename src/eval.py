import argparse 
import os, sys
import random
import numpy as np
import tensorflow as tf

from src.trainers.trainer import Trainer
from .utils import utils, ops_utils
from . import configs
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--mode', type=str, default="eval")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument('--metrics', type=str, default="wer")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument('--transfer', type="bool", const=True, nargs="?", default=False,
                        help="If model needs custom load.")

    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--output', type="bool", const=True, nargs="?", default=False)
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")


def load_model(sess, Model, hparams):
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.tables_initializer())

    if hparams.load:
        ckpt = os.path.join(hparams.out_dir, "csp.%s.ckpt" % hparams.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)

    if ckpt:
        if FLAGS.transfer:
            Model.load(sess, ckpt, FLAGS)
        else:
            saver_variables = tf.global_variables()
            var_list = {var.op.name: var for var in saver_variables}
            for var in Model.ignore_save_variables() + ["batch_size",
                    "eval_batch_size", 'Variable_1']:
                if var in var_list:
                    del var_list[var]
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, ckpt)


def eval(hparams, flags=None):
    tf.reset_default_graph()
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.EVAL
    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)
    hparams.batch_size = hparams.eval_batch_size

    with graph.as_default():
        #eval_writer = tf.summary.FileWriter(
        #    os.path.join(hparams.summaries_dir, "log_eval"))

        trainer = Trainer(hparams, Model, BatchedInput, mode)
        trainer.build_model()

        sess = tf.Session(graph=graph)
        load_model(sess, Model, hparams)
        trainer.init(sess)

        dlgids = []
        lers = []

        pbar = tqdm(total=trainer.data_size, ncols=100)
        pbar.set_description("Eval")
        fo = open(os.path.join(hparams.summaries_dir, "eval_ret.txt"), "w")
        utils.prepare_output_path(hparams)
        lers = {}
        while True:
            try:
                ids, ground_truth_labels, predicted_labels = trainer.eval(sess)
                utils.write_log(hparams, [str(ground_truth_labels)])
                
                #dlgids += list([str(id).split('/')[-2] for id in ids])
                for acc_id, (gt_labels, p_labels) in enumerate(zip(ground_truth_labels, predicted_labels)):
                    if acc_id not in lers: lers[acc_id] = []
                    for i in range(len(gt_labels)):
                        ler, str_original, str_decoded = ops_utils.evaluate(
                            gt_labels[i], p_labels[i],
                            trainer._batched_input_test.decode, 
                            flags.metrics, acc_id)
                        if ler is not None:
                            lers[acc_id].append(ler)
                            #filename = input_filenames[i].decode('utf-8')
                            filename = ""
                        
                            #if flags.verbose:
                            #if i == 0:
                            tqdm.write("\nGT: %s\nPR: %s\nLER: %.3f\n" % (' '.join(str_original), ' '.join(str_decoded), ler))
                            #fo.write("GT: %s\t%s\t%.3f\n" % (' '.join(str_original), ' '.join(str_decoded), ler))

                            meta = tf.SummaryMetadata()
                            meta.plugin_data.plugin_name = "text"
                            summary = tf.Summary()
                            #summary.value.add(
                            #    tag=os.path.basename(filename),
                            #    metadata=meta,
                            #    tensor=tf.make_tensor_proto('%s\n\n*(LER=%.3f)* ->\n\n%s' % (
                            #    ' '.join(str_original), sum(lers) / len(lers), ' '.join(str_decoded)), dtype=tf.string))
                            #eval_writer.add_summary(summary, trainer.epoch_exact)
                        else:
                            tqdm.write(str(gt_labels))

                # update pbar progress and postfix
                pbar.update(trainer.batch_size)
                bar_pf = {}
                for acc_id in range(len(ground_truth_labels)):
                    bar_pf["er" + str(acc_id)] = "%2.2f" % (sum(lers[acc_id]) / len(lers[acc_id]) * 100)
                pbar.set_postfix(bar_pf)
            except tf.errors.OutOfRangeError:
                break

    #acc_by_ids = {}
    #for i, id in enumerate(dlgids):
    #    if id not in acc_by_ids: acc_by_ids[id] = []
    #    acc_by_ids[id].append(lers[0][i])

    print("\n\n----- Statistics -----")
    for id, ls in acc_by_ids.items():
        print("%s\t%2.2f" % (id, sum(ls) / len(ls)))

    fo.write("LER: %2.2f" % (sum(lers) / len(lers) * 100))
    print(len(lers[0]))
    fo.close()

    tf.logging.info("test_ler = {:.3f}".format(sum(lers) / len(lers)))
    #eval_writer.add_summary(
    #    tf.Summary(value=[tf.Summary.Value(simple_value=sum(lers) / len(lers), tag="label_error_rate")]), trainer.epoch_exact)
    #if summary: eval_writer.add_summary(summary, trainer.epoch_exact)


def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = configs.HCOPY_CONFIG_PATH

    if not os.path.exists(hparams.summaries_dir): os.mkdir(hparams.summaries_dir)
    
    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    eval(hparams, FLAGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
