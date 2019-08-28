import sys
import cPickle as pickle
from collections import defaultdict


import numpy as np
# import pandas as pd
from gensim.corpora import Dictionary


min_key_freq = 5
max_sent_len = 64


def preprocess_dataset(f_train, f_dev, f_test):
    def doc2idx(d, row, max_len):
        # import pdb; pdb.set_trace()
        s_, k_, p_ = d.doc2idx(row[0]), row[1], row[2]
        for i in range(len(s_)):
            if s_[i] == -1:
                s_[i] = 1

        s_ = s_ + [0] * (max_sent_len - len(s_))
        return [s_, k_, p_]

    sentences = [["<pad>", "<unk>"]]

    s_data = [[], [], []]
    fs = [f_train, f_dev, f_test]
    for i, filename in enumerate(fs):
        with open(filename) as f:
            for line in f:
                try:
                    sent, sent_keys, pred_keys = line.strip().split("\t")
                except:
                    print("line:", line)
                    continue
                sent = sent.strip().split()
                sent_keys = sent_keys.strip().split()
                pred_keys = pred_keys.strip().split()
                sentences.append(sent)
                s_data[i].append((sent, sent_keys, pred_keys))

    dictionary = Dictionary(sentences)
    global max_sent_len
    data = [[doc2idx(dictionary, row, max_sent_len) for row in dt] for dt in s_data]

    ks = [r[2] for r in data[0]]
    ks+= [r[2] for r in data[1]]
    ks+= [r[2] for r in data[2]]

    keys = []
    for k in ks:
        keys.extend(k)
    keys = list(set(keys))

    keys_ = {}
    for i,k in enumerate(keys):
        keys_[k] = i

    for dt in data:
        for row in dt:
            row[2] = [keys_[r] for r in row[2]]

    train_, dev_, test_ = data
    num_classes = len(keys)

    return train_, dev_, test_, dictionary, num_classes


def preprocess_data_weibo_base(out):
    train_, dev_, test_, dictionary, num_classes = preprocess_dataset(f_train="data/train.weibo.txt",
                   f_dev="data/valid.weibo.txt",
                   f_test="data/test.weibo.txt")
    cached = {
        "train": train_,
        "dev": dev_,
        "test": test_,
        "dictionary": dictionary,
        "vocab_size": len(dictionary),
        "w2v": np.random.randn(len(dictionary), 300),
        "max_sent": max_sent_len,
        "num_classes": num_classes
    }

    pickle.dump(cached, open(out, "wb+"))

def preprocess_data_weibo_typeA(weibo, out):
    weibo = pickle.load(open(weibo, "rb"))
    train_ = weibo["train"]
    dev_ = weibo["dev"]
    test_ = weibo["test"]

    X = [[], [], []]
    L = [[], [], []]

    for i, data in enumerate([train_, dev_, test_]):
        for row in data:
            sent = row[0]
            labels = row[2]
            X[i].append(sent)
            L[i].append(labels)

    w2v = weibo["w2v"]
    num_classes = weibo["num_classes"]
    cached = {
        "train": X[0],
        "train_label": L[0],
        "dev": X[1],
        "dev_label": L[1],
        "test": X[2],
        "test_label": L[2],
        "num_classes": num_classes,
        "w2v": w2v
    }
    pickle.dump(cached, open(out, "wb+"))


def load_data_weibo_typeA(weibo):
    weibo = pickle.load(open(weibo, "rb"))
    train = weibo["train"]
    train_label = weibo["train_label"]
    test = weibo["test"]
    test_label = weibo["test_label"]
    dev = weibo["dev"]
    dev_label = weibo["dev_label"]
    w2v = weibo["w2v"]
    num_classes = weibo["num_classes"]
    return train, train_label, test, test_label, dev, dev_label, w2v, num_classes

if __name__ == "__main__":
    if sys.argv[1] == "base":
        preprocess_data_weibo_base(sys.argv[2])
    elif sys.argv[1] == "typeA":
        preprocess_data_weibo_typeA(sys.argv[2], sys.argv[3])
