import os
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
from python_speech_features import *
import numpy as np

DATA_DIR = os.path.join("data", "vivos")

if __name__ == "__main__":
    mode = "train"

    wset = set()
    with open(os.path.join(DATA_DIR, mode, "prompts.txt"), encoding="utf-8") as f:
        for s in f.read().split('\n'):
            words = s.split(' ')[1:]
            for word in words: wset.add(word.lower())

    print(len(wset))
    with open(os.path.join(DATA_DIR, mode, "words.txt"), "w", encoding="utf-8") as f:
        f.write("0 <unk>\n1 <sos>\n2 <eos>\n")
        f.write("\n".join(["%d %s" % (i + 2, word) for i, word in
            enumerate(list(wset)) if word != ""]))

