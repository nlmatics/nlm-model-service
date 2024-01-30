import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

from .tree import tree

# from theano import config


def getWordmap(file_path):
    if file_path.endswith("txt"):
        words = {}
        w_e = []
        f = open(file_path)
        lines = f.readlines()
        for (n, i) in enumerate(lines):
            i = i.split()
            j = 1
            v = []
            while j < len(i):
                v.append(float(i[j]))
                j += 1
            words[i[0]] = n
            w_e.append(v)
        return words, np.array(w_e)
    elif file_path.endswith("h5"):
        reg_vectors_df = pd.read_hdf(file_path)
        reg_index = reg_vectors_df.index.values
        words = {word: idx for idx, word in enumerate(reg_index)}
        w_e = reg_vectors_df.values
        return words, w_e


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype("int32")
    x_mask = np.zeros((n_samples, maxlen)).astype("float32")
    for idx, s in enumerate(list_of_seqs):
        x[idx, : lengths[idx]] = s
        x_mask[idx, : lengths[idx]] = 1.0
    x_mask = np.asarray(x_mask, dtype="float32")
    return x, x_mask


def lookupIDX(words, w):
    w = w.lower()
    if len(w) > 1 and w[0] == "#":
        w = w.replace("#", "")
    if len(w) > 1 and (w[0] == "$" or w[-1] == "$"):
        w = w.replace("$", "")
    if w in words:
        return words[w]
    else:
        return len(words) - 1


def getSeq(p1, words):
    # p1 = p1.split()
    p1 = wordpunct_tokenize(p1)
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def getSeqs(p1, p2, words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    for i in p2:
        X2.append(lookupIDX(words, i))
    return X1, X2


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def getSimEntDataset(f, words, task):
    data = open(f)
    lines = data.readlines()
    examples = []
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split("\t")
            if len(i) == 3:
                if task == "sim":
                    e = (tree(i[0], words), tree(i[1], words), float(i[2]))
                    examples.append(e)
                elif task == "ent":
                    e = (tree(i[0], words), tree(i[1], words), i[2])
                    examples.append(e)
                else:
                    raise ValueError("Params.traintype not set correctly.")

            else:
                pass
    return examples


def getSentimentDataset(f, words):
    data = open(f)
    lines = data.readlines()
    examples = []
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split("\t")
            if len(i) == 2:
                e = (tree(i[0], words), i[1])
                examples.append(e)
            else:
                pass
    return examples


def getDataSim(batch, nout):
    g1 = []
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    if nout <= 0:
        return (scores, g1x, g1mask, g2x, g2mask)

    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype="float32")
    return (scores, g1x, g1mask, g2x, g2mask)


def getDataEntailment(batch):
    g1 = []
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(3)
        label = i[2].strip()
        if label == "CONTRADICTION":
            temp[0] = 1
        if label == "NEUTRAL":
            temp[1] = 1
        if label == "ENTAILMENT":
            temp[2] = 1
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype="float32")
    return (scores, g1x, g1mask, g2x, g2mask)


def getDataSentiment(batch):
    g1 = []
    for i in batch:
        g1.append(i[0].embeddings)

    g1x, g1mask = prepare_data(g1)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[1].strip()
        if label == "0":
            temp[0] = 1
        if label == "1":
            temp[1] = 1
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype="float32")
    return (scores, g1x, g1mask)


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def sentiment2idx(sentiment_file, words):
    """
    Read sentiment data file, output array of word indices that can be fed into the algorithms.
    :param sentiment_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, golds. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location), golds[i] is the label (0 or 1) for sentence i.
    """
    f = open(sentiment_file)
    lines = f.readlines()
    golds = []
    seq1 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]
        score = int(i[1])  # score are labels 0 and 1
        X1 = getSeq(p1, words)
        seq1.append(X1)
        golds.append(score)
    x1, m1 = prepare_data(seq1)
    return x1, m1, golds


def sim2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the score for pair i (float). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file)
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]
        p2 = i[1]
        score = float(i[2])
        X1, X2 = getSeqs(p1, p2, words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1, m1 = prepare_data(seq1)
    x2, m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds


def entailment2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the label for pair i (CONTRADICTION NEUTRAL ENTAILMENT). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file)
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]
        p2 = i[1]
        score = i[2]
        X1, X2 = getSeqs(p1, p2, words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1, m1 = prepare_data(seq1)
    x2, m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds


def getWordWeight(local_weightfile, global_weightfile, a=1e-3):
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0
    word2weight = {}
    local_word2weight = {}
    local_tfidf = {}
    if local_weightfile:
        with open(local_weightfile) as f:
            lines = f.readlines()
        N = 0
        for i in lines:
            i = i.strip()
            if len(i) > 0:
                i = i.split()
                if len(i) == 3:
                    local_word2weight[i[0]] = float(i[1])
                    local_tfidf[i[0]] = float(i[1]) * float(i[2])
                else:
                    pass
    global_word2weight = {}
    with open(global_weightfile) as f:
        lines = f.readlines()
        N = 0
        for i in lines:
            i = i.strip()
            if len(i) > 0:
                i = i.split()
                if len(i) == 2:
                    global_word2weight[i[0]] = float(i[1])
                    N += float(i[1])
                else:
                    pass
    for key, freq in global_word2weight.items():
        # p_l = local_word2weight.get(key, 1)
        word2weight[key] = a / (a + (freq / N) ** 2)
        # word2weight[key] *= 0.0001 / (0.0001 + p_l)
    return word2weight


def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            # for OOV
            weight4ind[ind] = 0.01
    return weight4ind


def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype("float32")
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype="float32")
    return weight
