import argparse
import os
import sys
import wave
from array import array

import pyaudio
import tensorflow as tf

from . import configs
from .utils import utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--config', type=str, default=None)


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
        self.batched_input.reset_iterator(sess)
        with self.graph.as_default():
            while True:
                try:
                    sample_ids, _ = self.model.infer(sess)

                    for i in range(len(sample_ids)):
                        # str_original = BatchedInput.decode(target_labels[i])
                        str_decoded = self.batched_input.decode(sample_ids[i])

                        # print('Original: %s' % str_original)
                        print(' -> Result:\n\t\t%s' % "".join(str_decoded))
                except tf.errors.OutOfRangeError:
                    return


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


def record(filename):
    FORMAT = pyaudio.paInt16
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    count_silent = 0
    count_voice = 0
    while True:
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        vol = max(data_chunk)
        if vol >= 500:
            print("<voice>                 ", end='\r')
            count_silent = 0
            count_voice += 1
        else:
            print("<silent>                ", end='\r')
            count_silent += 1

        if count_silent > 20:
            if count_voice > 10: break
            else:
                frames = []
                count_silent = 0
                count_voice = 0
        else:
            frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # writing to file
    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(1)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))  # append frames recorded to file
    wavfile.close()


def play(filename):
    CHUNK = 1024
    wf = wave.open(filename, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(CHUNK)

    # play stream (looping from beginning of file to the end)
    while data != b'':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(CHUNK)

    # cleanup stuff.
    stream.close()
    p.terminate()


def main(unused_argv):
    hparams = utils.create_hparams(FLAGS)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = configs.HCOPY_CONFIG_PATH

    hparams.input_path = os.path.join("tmp", "input.tmp")
    BatchedInput = utils.get_batched_input_class(hparams)
    Model = utils.get_model_class(hparams)
    sess, model, _ = load(Model, BatchedInput, hparams)

    os.system('cls')

    while True:
        #if input("Start recording? [Y/n]: ") != 'n':
        print("Recording...", end="\r")
        record("test.wav")
        # infer(hparams)
        # record()
        print("Inferring...", end="\r")
        model.infer("test.wav", sess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
