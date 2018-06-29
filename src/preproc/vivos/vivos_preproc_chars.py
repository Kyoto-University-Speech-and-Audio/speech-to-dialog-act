import os
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
from python_speech_features import *
import numpy as np

DATA_DIR = os.path.join("data", "vivos")

if __name__ == "__main__":
    mode = "train"

    d = {}
    chars = open(os.path.join(DATA_DIR, "chars.txt")).read().split('\n')
    chars = { char.split(' ')[1]: char.split(' ')[0] for char in chars if char != ""}
    chars[' '] = '2'
    outputs = []
    with open(os.path.join(DATA_DIR, mode, "prompts.txt"), encoding="utf-8") as f:
        lines = f.read().split("\n")
        for i, s in enumerate(lines):
            filename = s.split(' ')[0]
            if filename == "": continue
            d[filename] = ' '.join(s.split(' ')[1:])

            filename = os.path.join("data", "vivos", mode, "feature", filename)

            s = ' '.join(s.split(' ')[1:])
            outputs.append(
                (filename,
                ' '.join(['1'] + [chars[c.lower()] if c.lower() in chars else \
                        c for c in list(s)] + ['0']))
            )
            
            if i % 100 == 1: print("%d / %d" % (i, len(lines)))

    outputs.sort(key=lambda x: len(x[1]))
    with open(os.path.join(DATA_DIR, mode, "data_chars.txt"), "w", encoding="utf-8") as f:
        f.write('\n'.join(["%s %s" % (fn, tg) for fn, tg in outputs]))
        
