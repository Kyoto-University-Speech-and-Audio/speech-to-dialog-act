import argparse
import os
import tensorflow as tf
import sys
from .utils import utils
import pyaudio
import wave
from array import array
import os

FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
CHUNK=1024
RECORD_SECONDS=15
FILE_NAME="RECORDING.wav"

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="vivos")
    parser.add_argument('--model', type=str, default="ctc")
    parser.add_argument('--input_unit', type=str, default="char", help="word | char")
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--summaries_dir", type=str, default="log")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    parser.add_argument('--server', type="bool", const=True, nargs="?", default=False)


class ModelWrapper:
    def __init__(self, hparams, mode, BatchedInput, Model):
        self.graph = tf.Graph()
        self.hparams = hparams
        self.BatchedInput = BatchedInput
        self.mode = mode

        with self.graph.as_default():
            self.batched_input = BatchedInput(hparams, mode)
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
            return global_step

    def infer(self, wavfile, sess):
        self.hparams.input_path = os.path.join("tmp", "input.tmp")
        with open(self.hparams.input_path, "w") as f:
            f.write(wavfile)
        self.batched_input = self.BatchedInput(self.hparams, self.mode)
        with self.graph.as_default():
            while True:
                try:
                    sample_ids, _ = self.model.infer(sess)

                    for i in range(len(sample_ids)):
                        # str_original = BatchedInput.decode(target_labels[i])
                        str_decoded = self.batched_input.decode(sample_ids[i])

                        # print('Original: %s' % str_original)
                        print('Decoded:  %s' % "".join(str_decoded))
                except tf.errors.OutOfRangeError:
                    break

def load(Model, BatchedInput, hparams):
    infer_model = ModelWrapper(
        hparams,
        tf.estimator.ModeKeys.PREDICT,
        BatchedInput, Model
    )

    infer_sess = tf.Session(graph=infer_model.graph)

    with infer_model.graph.as_default():
        global_step = infer_model.load_model(
            infer_sess, "infer"
        )
        hparams.input_path = "none"
        infer_model.batched_input.reset_iterator(infer_sess)

    return infer_sess, infer_model, global_step

def record():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        vol = max(data_chunk)
        if vol >= 500:
            print("something said", end='\r')
            frames.append(data)
        else:
            print("nothing", end='\r')

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # writing to file
    wavfile = wave.open(FILE_NAME, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))  # append frames recorded to file
    wavfile.close()

def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)

    os.system('cls')
    hparams.input_path = os.path.join("tmp", "input.tmp")
    with open(hparams.input_path, "w") as f: f.write("empty")
    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)
    sess, model, _ = load(Model, BatchedInput, hparams)

    # infer(hparams)
    # record()
    model.infer("test.wav", sess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
