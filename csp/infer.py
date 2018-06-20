import argparse
import os
import tensorflow as tf
import sys

from .models.base import BaseModelWrapper
from .utils import utils, ops_utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('test')

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size.")

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    parser.add_argument("--target_path", type=str, default=None)

    parser.add_argument('--server', type="bool", const=True, nargs="?", default=False)


class ModelWrapper(BaseModelWrapper):
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        with self.graph.as_default():
            self.batched_input = BatchedInput(hparams, mode)
            self.batched_input.init_dataset()
            self.iterator = self.batched_input.iterator
            self.model = Model(
                hparams,
                mode=mode,
                iterator=self.iterator
            )

def load(Model, BatchedInput, hparams):
    infer_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.PREDICT,
        BatchedInput, Model
    )

    infer_sess = tf.Session(graph=infer_model.graph)

    with infer_model.graph.as_default():
        _, global_step = infer_model.load_model(
            infer_sess, FLAGS.load
        )

        infer_model.batched_input.reset_iterator(infer_sess)

    return infer_sess, infer_model, global_step


def infer(hparams):
    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)
    infer_sess, infer_model, global_step = load(Model, BatchedInput, hparams)
    writer = tf.summary.FileWriter(
        os.path.join(hparams.summaries_dir, "%s_%s" % (hparams.model, hparams.dataset), "log_infer"),
        infer_sess.graph)

    targets = []
    if FLAGS.target_path:
        targets = open(FLAGS.target_path).read().split('\n')

    count = 0
    lers = []
    with infer_model.graph.as_default():
        while True:
            try:
                sample_ids, summary = infer_model.model.infer(infer_sess)
                if summary:
                    writer.add_summary(summary, global_step)

                for i in range(len(sample_ids)):
                    str_decoded = infer_model.batched_input.decode(sample_ids[i])

                    if FLAGS.target_path:
                        print('-- Original: %s' % targets[count])

                    print('   Decoded:  %s' % "".join(str_decoded))

                    if FLAGS.target_path:
                        sdecoded = "".join(str_decoded).replace('<sp>', '')
                        starget = targets[count].replace(' ', '')
                        ler = ops_utils.levenshtein(list(sdecoded), list(starget)) / len(starget)
                        if ler > 1: ler = 1
                        print('   LER:      %.3f' % ler)
                        lers.append(ler)

                    count += 1
            except tf.errors.OutOfRangeError:
                break

    print("LER: %2.2f" % (sum(lers) / count * 100))

def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = "/n/sd7/trung/bin/htk/HTKTools/HCopy"
    # hparams.hcopy_path = os.path.join("bin", "htk", "bin.win32", "HCopy.exe")
    hparams.hcopy_config = os.path.join("/n/sd7/trung/config.lmfb.40ch")
    # hparams.hcopy_config = os.path.join("data", "config.lmfb.40ch")

    print(FLAGS.server)
    if FLAGS.server:
        from flask import Flask
        app = Flask(__name__)

        @app.route("/")
        def hello():
            return "Hello"
    else:
        infer(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
