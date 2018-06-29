import os
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
from python_speech_features import *
import numpy as np

DATA_DIR = os.path.join("data", "vivos")

if __name__ == "__main__":
    cset = set()
    with open(os.path.join(DATA_DIR, "train", "prompts.txt"), encoding="utf-8") as f:
        for s in f.read().split('\n'):
            chars = list(s)
            for char in chars: cset.add(char.lower())
    
    with open(os.path.join(DATA_DIR, "test", "prompts.txt"), encoding="utf-8") as f:
        for s in f.read().split('\n'):
            chars = list(s)
            for char in chars: cset.add(char.lower())

    cset.remove(' ')
    print(len(cset))
    with open(os.path.join(DATA_DIR, "chars.txt"), "w", encoding="utf-8") as f:
        f.write("0 <eos>\n1 <sos>\n2 <sp>\n")
        f.write("\n".join(["%d %s" % (i + 3, char) for i, char in
            enumerate(list(cset)) if char != ""]))

