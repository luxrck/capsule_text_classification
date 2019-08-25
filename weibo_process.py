import sys
import cPickle as pickle
from collections import defaultdict


import numpy as np
# import pandas as pd
from gensim.corpora import Dictionary


min_key_freq = 5
max_sent_len = 64


def preprocess_dataset(f_train, f_dev, f_test):
    def keys(keywords):
        # import pdb; pdb.set_trace()
        d = defaultdict(int)
        for ks in keywords:
            for k in ks:
                d[k] += 1
        return d

    def doc2idx(d, row, max_len):
        # import pdb; pdb.set_trace()
        s_, k_, p_ = d.doc2idx(row[0]), d.doc2idx(row[1]), d.doc2idx(row[2])
        for i in range(len(s_)):
            if s_[i] == -1:
                s_[i] = 1

        s_ = s_ + [0] * (max_sent_len - len(s_))
        return [s_, k_, p_]

    sentences = [["<pad>", "<unk>"]]

    max_sent_len = 2

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
                sentences.append(sent + sent_keys + pred_keys)
                s_data[i].append((sent, sent_keys, pred_keys))
                if len(sent) > max_sent_len:
                    max_sent_len = len(sent)

    dictionary = Dictionary(sentences)
    data = [[doc2idx(dictionary, row, max_sent_len) for row in dt] for dt in s_data]

    import pdb; pdb.set_trace()
    ks = [r[2] for r in data[0]]
    ks+= [r[2] for r in data[1]]
    ks+= [r[2] for r in data[2]]
    keys_counted = keys(ks)
    keys_sorted = sorted(keys_counted.items(), key=lambda k: k[1], reverse=True)

    next_avaliable_class = 1     # 0 is for keys which its key_freq < min_key_freq
    keys_ = defaultdict(int)
    for k,c in keys_sorted:
        if c < min_key_freq:
            keys_[k] = 0
            continue
        keys_[k] = next_avaliable_class
        next_avaliable_class += 1

    for dt in data:
        for row in dt:
            row[2] = [keys_[r] for r in row[2]]

    train_, dev_, test = data
    num_classes = next_avaliable_class

    import pdb; pdb.set_trace()
    return train_, dev_, test, dictionary, max_sent_len, num_classes


def preprocess_data_weibo_base(out):
    train_, dev_, test, dictionary, max_sent_len, num_classes = preprocess_dataset(f_train="data/train.weibo.txt",
                   f_dev="data/valid.weibo.txt",
                   f_test="data/test.weibo.txt")
    cached = {
        "train": train_,
        "dev": dev_,
        "test": test,
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
    train, train_label = [], []
    for row in train_:
        labels = row[-1]
        train.extend([row[0] for _ in range(len(labels))])
        train_label.extend(labels)

    dev_ = weibo["dev"]
    dev, dev_label = [], []
    for row in dev_:
        labels = row[-1]
        dev.extend([row[0] for _ in range(len(labels))])
        dev_label.extend(labels)

    test_ = weibo["test"]
    test, test_label = [], []
    for row in test_:
        labels = row[-1]
        test.extend([row[0] for _ in range(len(labels))])
        test_label.extend(labels)

    w2v = weibo["w2v"]
    num_classes = weibo["num_classes"]
    cached = {
        "train": train,
        "train_label": train_label,
        "test": test,
        "test_label": test_label,
        "dev": dev,
        "dev_label": dev_label,
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