import os
import numpy as np

class WordEmbedding():
    def __init__(self):
        data_path = os.path.join("data", "word2vec", "googlenews.bin")
        print("Load word2vec file {}\n".format(data_path))
        with open(data_path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            word2vec = {}
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1).decode('utf-8', errors='ignore')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                print(len(word2vec))
                if word != '':
                    word2vec[word] = np.fromstring(f.read(binary_len), dtype='float32')
                    # print(word, word2vec[word])
                else:
                    f.read(binary_len)

if __name__ == "__main__":
    embedding = WordEmbedding()