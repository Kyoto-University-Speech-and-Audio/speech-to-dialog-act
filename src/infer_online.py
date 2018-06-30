import argparse
import os
import sys
import wave
from array import array

import pyaudio
import tensorflow as tf

from src.trainers.trainer import Trainer
from . import configs
from .utils import utils

sys.path.insert(0, os.path.abspath('.'))
tf.logging.set_verbosity(tf.logging.INFO)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--config', type=str, default=None)


def infer(wavfile, sess, model, batched_input, hparams):
    hparams.input_path = os.path.join("tmp", "input.tmp")
    with open(hparams.input_path, "w") as f:
        f.write(wavfile)
    batched_input.reset_iterator(sess)
    with graph.as_default():
        while True:
            try:
                sample_ids, _ = model.infer(sess)

                for i in range(len(sample_ids)):
                    # str_original = BatchedInput.decode(target_labels[i])
                    str_decoded = batched_input.decode(sample_ids[i])

                    # print('Original: %s' % str_original)
                    print(' -> Result:\n\t\t%s' % "".join(str_decoded))
            except tf.errors.OutOfRangeError:
                return


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
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

def load(Model, BatchedInput, hparams):
    tf.reset_default_graph()
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.PREDICT

    with graph.as_default():
        batched_input = BatchedInput(hparams, mode)
        batched_input.init_dataset()

        trainer = Trainer(hparams, Model, BatchedInput, mode)
        trainer.build_model()

        sess = tf.Session(graph=graph)
        load_model(sess, Model, hparams)
        trainer.init(sess)

        batched_input.reset_iterator(sess)

        hparams.input_path = "none"
        batched_input.reset_iterator(sess)

    return sess, trainer, batched_input


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
    sess, trainer, batched_input = load(Model, BatchedInput, hparams)

    os.system('cls')

    while True:
        #if input("Start recording? [Y/n]: ") != 'n':
        print("Recording...", end="\r")
        record("test.wav")
        # infer(hparams)
        # record()
        print("Inferring...", end="\r")
        infer("test.wav", sess, trainer.eval_model, batched_input, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
