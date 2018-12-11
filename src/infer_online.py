import argparse
import os
import sys
import wave
from array import array
import json

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
    parser.add_argument('--load', type=str, default=None)


def infer(wav_file, sess, trainer):
    with open(trainer.hparams.input_path, "w") as f:
        f.write("sound\n")
        f.write(wav_file + "\n")

    with sess.graph.as_default():
        trainer.infer(sess)


def load_model(sess, Model, hparams):
    sess.run(tf.global_variables_initializer())

    if hparams.load:
        ckpt = hparams.load or os.path.join(hparams.out_dir, "csp.%s.ckpt" % hparams.load)
    else:
        ckpt = tf.train.latest_checkpoint(hparams.out_dir)

    if ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)


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
            if count_voice > 10:
                break
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
    chunk = 1024
    wf = wave.open(filename, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != b'':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    stream.close()
    p.terminate()


def main(unused_argv):
    if args.config is None:
        raise Exception("Config file must be provided")

    json_file = open('model_configs/%s.json' % args.config).read()
    json_dict = json.loads(json_file)
    input_cls = utils.get_batched_input_class(json_dict.get("dataset", "default"))
    model_cls = utils.get_model_class(json_dict.get("model"))
    print(input_cls)

    hparams = utils.create_hparams(args, model_cls)
    hparams.hcopy_path = configs.HCOPY_PATH
    hparams.hcopy_config = configs.HCOPY_CONFIG_PATH

    hparams.input_path = os.path.join("tmp", "input.tmp")

    tf.reset_default_graph()
    graph = tf.Graph()
    mode = tf.estimator.ModeKeys.PREDICT

    with graph.as_default():
        trainer = Trainer(hparams, model_cls, input_cls, mode)
        trainer.build_model()

        sess = tf.Session(graph=graph)
        load_model(sess, model_cls, hparams)
        trainer.init(sess)
    
    os.system('cls')  # clear screen

    infer("test.wav", sess, trainer)
    return
    while True:
        # if input("Start recording? [Y/n]: ") != 'n':
        print("Recording...", end="\r")
        record("test.wav")
        # infer(hparams)
        # record()
        print("Inferring...", end="\r")
        infer("test.wav", sess, trainer)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    add_arguments(_parser)
    args, unparsed = _parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
