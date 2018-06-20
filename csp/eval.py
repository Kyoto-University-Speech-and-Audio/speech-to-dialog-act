import argparse 
import os, sys
import random
import numpy as np
import tensorflow as tf
from .utils import utils, ops_utils
from .models.base import BaseModelWrapper
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")

    parser.add_argument('--load', type=str, default=None)
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


class ModelWrapper(BaseModelWrapper):
    def __init__(self, hparams, mode, BatchedInput, Model):
        super().__init__()
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
            eval_sess, hparams.load
        )

        eval_model.batched_input.reset_iterator(eval_sess)
        test_ler = 0
        total_count = 0

    pbar = tqdm(total=eval_model.batched_input.size, ncols=100)
    pbar.set_description("Eval")
    fo = open(os.path.join(hparams.summaries_dir, "eval_ret_%d.txt" % global_step), "w")
    while True:
        try:
            input_filenames, target_labels, _, decoded, summary = loaded_eval_model.eval(eval_sess)

            for i in range(len(target_labels)):
                str_original = eval_model.batched_input.decode(target_labels[i])
                str_decoded = eval_model.batched_input.decode(decoded[i])
                str_original = list(filter(lambda it: it != '<sp>', str_original))
                str_decoded = list(filter(lambda  it: it != '<sp>', str_decoded))
                if len(str_original) != 0:
                    # ler = ops_utils.levenshtein(''.join(str_original), ''.join(str_decoded)) / len(''.join(str_original))
                    # ler = ops_utils.levenshtein(target_labels[i], decoded[i]) / len(target_labels[i])
                    ler = ops_utils.levenshtein(str_original, str_decoded) / len(str_original)
                else: continue
                test_ler += ler
                total_count += 1
                filename = input_filenames[i].decode('utf-8')

                fo.write("%s\t%s\t%.3f\n" % (filename, ' '.join(str_decoded), ler))

                meta = tf.SummaryMetadata()
                meta.plugin_data.plugin_name = "text"
                summary = tf.Summary()
                summary.value.add(
                    tag=os.path.basename(filename),
                    metadata=meta,
                    tensor=tf.make_tensor_proto('%s\n\n*(LER=%.3f)* ->\n\n%s' % (
                    ' '.join(str_original), test_ler / total_count, ' '.join(str_decoded)), dtype=tf.string))
                eval_writer.add_summary(summary, global_step)

            pbar.update(hparams.batch_size)
            pbar.set_postfix(ler="%.3f" % (test_ler / total_count))
        except tf.errors.OutOfRangeError:
            break

    fo.write("LER: %.3f" % (test_ler / total_count))
    fo.close()

    tf.logging.info("test_ler = {:.3f}".format(test_ler / total_count))
    eval_writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(simple_value=test_ler / total_count, tag="label_error_rate")]), global_step)
    if summary: eval_writer.add_summary(summary, global_step)
    return test_ler / total_count


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

    eval(hparams, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
