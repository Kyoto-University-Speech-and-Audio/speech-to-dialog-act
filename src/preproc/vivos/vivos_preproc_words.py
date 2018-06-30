import os
from tensorflow.python.platform import gfile
import scipy.io.wavfile as wav
from python_speech_features import *
import numpy as np
from subprocess import call

DATA_DIR = os.path.join("data", "vivos")

if __name__ == "__main__":
    mode = "test"

    d = {}
    words = open(os.path.join(DATA_DIR, "words.txt")).read().split('\n')
    words = { word.split(' ')[1]: word.split(' ')[0] for word in words if word != ""}
    outputs = []
    with open(os.path.join(DATA_DIR, mode, "prompts.txt"), encoding="utf-8") as f:
        lines = f.read().split("\n")
        for i, s in enumerate(lines):
            filename = s.split(' ')[0]
            if filename == "": continue
            d[filename] = ' '.join(s.split(' ')[1:])
            wav_filename = os.path.join("data", "vivos", mode, "waves", filename.split('_')[0], filename + ".wav")
            (rate, sig) = wav.read(wav_filename)

            # mfcc_feat = mfcc(sig, rate, numcep=40, nfilt=40)
            # d_mfcc_feat = delta(mfcc_feat, 1)
            # d2_mfcc_feat = delta(d_mfcc_feat, 1)

            # fbank_feat = logfbank(sig, rate)
            # d_fbank_feat = delta(fbank_feat, 1)
            # d2_fbank_feat = delta(d_fbank_feat, 1)
            
            # feat = [np.concatenate([x, y, z]).astype(np.float32) for x, y, z in zip(mfcc_feat, d_mfcc_feat, d2_mfcc_feat)]
            # feat = fbank_feat.astype(np.float32)
            # feat = [np.concatenate([x, y, z]) for x, y, z in zip(fbank_feat, d_fbank_feat, d2_fbank_feat)]
            filename = os.path.join("data", "vivos", mode, "feature", filename + ".htk")
            # np.save(filename, feat)
            call([
                "/n/sd7/trung/bin/htk/HTKTools/HCopy",
                wav_filename,
                filename,
                "-C", "/n/sd7/trung/config.lmfb.40ch"
            ])

            outputs.append(
                (filename,
                ' '.join(['2'] + [words[w.lower()] if w.lower() in words else \
                    '0' for w in s.split(' ')[1:]] + ['1']))
            )
            
            if i % 100 == 1: print("%d / %d" % (i, len(lines)))

    outputs.sort(key=lambda x: len(x[1]))
    with open(os.path.join(DATA_DIR, mode, "data.txt"), "w", encoding="utf-8") as f:
        f.write('\n'.join(["%s %s" % (fn, tg) for fn, tg in outputs]))
        
