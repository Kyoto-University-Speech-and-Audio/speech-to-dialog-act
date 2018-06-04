import argparse 
import os, sys
import random
import numpy as np
import tensorflow as tf
from .utils import utils, ops_utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('test')

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")

    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

    parser.add_argument('--sample_rate', type=float, default=16000)
    parser.add_argument('--window_size_ms', type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")


class ModelWrapper:
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        with self.graph.as_default():
            self.batched_input = BatchedInput(hparams, mode)
            hparams.num_classes = self.batched_input.num_classes
            self.batched_input.init_dataset()
            self.iterator = self.batched_input.iterator
            self.model = Model(
                hparams,
                mode=mode,
                iterator=self.iterator
            )

    def load_model(self, sess, name):
        latest_ckpt = tf.train.latest_checkpoint(self.hparams.out_dir)
        if latest_ckpt:
            self.model.saver.restore(sess, latest_ckpt)
            sess.run(tf.tables_initializer())
            global_step = self.model.global_step.eval(session=sess)
            return self.model, global_step


def eval(hparams, output=True):
    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)

    eval_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.EVAL,
        BatchedInput, Model
    )

    eval_writer = tf.summary.FileWriter(
        os.path.join(hparams.summaries_dir, "log_eval"))

    eval_sess = tf.Session(graph=eval_model.graph)

    with eval_model.graph.as_default():
        loaded_eval_model, global_step = eval_model.load_model(
            eval_sess, "eval"
        )

        eval_model.batched_input.reset_iterator(eval_sess)
        test_ler = 0
        total_count = 0
        
    while True:
        try:
            target_labels, _, decoded = loaded_eval_model.eval(eval_sess)

            for i in range(len(target_labels)):
                str_original = eval_model.batched_input.decode(target_labels[i])
                str_decoded = eval_model.batched_input.decode(decoded[i])
                if len(str_original) != 0:
                    ler = ops_utils.levenshtein(str_original, str_decoded) / len(str_original)
                else: continue
                test_ler += ler
                total_count += 1
                if output:
                    print('-- Original: %s' % ''.join(str_original))
                    print('   Decoded:  %s' % ''.join(str_decoded))
                    print('   LER:      %.3f' % ler)

        except tf.errors.OutOfRangeError:
            break

    eval_writer.add_summary(eval_model.model.summary, global_step)
    eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(simple_value=test_ler / total_count, tag="label_error_rate")]), global_step)

    tf.logging.info("test_ler = {:.3f}".format(test_ler / total_count))
    return test_ler / total_count


def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    
    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    eval(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
