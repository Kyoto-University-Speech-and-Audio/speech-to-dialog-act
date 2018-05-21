import argparse
import os
import tensorflow as tf
import random
import numpy as np
import importlib
import sys
from .utils import utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('test')

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_encoder_layers", type=int, default=2,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=2,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")

    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")

    parser.add_argument('--sample_rate', type=float, default=16000)
    parser.add_argument('--window_size_ms', type=float, default=30.0)
    parser.add_argument('--window_stride_ms', type=float, default=10.0)

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

def create_hparams(flags):
    return tf.contrib.training.HParams(
        model=flags.model,
        dataset=flags.dataset,

        num_units=flags.num_units,
        num_encoder_layers=flags.num_encoder_layers,
        num_decoder_layers=flags.num_decoder_layers,
        summaries_dir=flags.summaries_dir,
        out_dir=flags.out_dir or "saved_models/%s_%s" % (flags.model, flags.dataset),
        num_train_steps=flags.num_train_steps,

        num_buckets=flags.num_buckets,

        sample_rate=flags.sample_rate,
        window_size_ms=flags.window_size_ms,
        window_stride_ms=flags.window_stride_ms,

        epoch_step=0,
    )

class ModelWrapper:
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        with self.graph.as_default():
            self.batched_input = BatchedInput(hparams, mode)
            self.hparams.batch_size = self.batched_input.size
            self.batched_input.init_dataset()
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

    def load_model(self, sess, name):
        latest_ckpt = tf.train.latest_checkpoint(self.hparams.out_dir)
        if latest_ckpt:
            self.model.saver.restore(sess, latest_ckpt)
            sess.run(tf.tables_initializer())
            return self.model

def infer(Model, BatchedInput, hparams):
    hparams.num_classes = BatchedInput.num_classes
    infer_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.PREDICT,
        BatchedInput, Model
    )

    infer_sess = tf.Session(graph=infer_model.graph)

    with infer_model.graph.as_default():
        loaded_infer_model = infer_model.load_model(
            infer_sess, "infer"
        )

        infer_model.batched_input.reset_iterator(infer_sess)
        sample_ids = loaded_infer_model.infer(infer_sess)

        # decoded_txt =
        # tf.summary.text('decoded_text', tf.py_func)

    for i in range(len(sample_ids[0])):
        # str_original = BatchedInput.decode(target_labels[i])
        str_decoded = BatchedInput.decode(sample_ids[0][i])

        # print('Original: %s' % str_original)
        print('Decoded:  %s' % str_decoded)

    # tf.logging.info("test_cost = {:.3f}, test_ler = {:.3f}".format(test_cost, test_ler))

def main(unused_argv):
    hparams = create_hparams(FLAGS)

    random_seed = FLAGS.random_seed
    if random_seed is not None and random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    BatchedInput = utils.get_batched_input_class(FLAGS)
    Model = utils.get_model_class(FLAGS)

    infer(Model, BatchedInput, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
