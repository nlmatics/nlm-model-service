import logging

import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def to_unicode(text, encoding="utf8", errors="strict"):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


def to_utf8(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf8")


def write_binary_embeddings(filename, embs, idx2word):
    with open(filename, "wb") as fout:
        fout.write(to_utf8("%s %s\n" % embs.shape))
        for idx, item in enumerate(idx2word):
            if idx % 10000 == 0:
                logger.info(f"Saving binary embeddings: {idx} / {embs.shape[0]}")
            fout.write(to_utf8(item) + b" " + embs[idx].astype(np.float32).tostring())


def read_binary_embeddings(filename, idx2word=[], word2idx={}):

    with open(filename, "rb") as fin:
        logger.info(f"loading from {filename}")

        line = to_unicode(fin.readline())
        voc_size, dimension = [int(x) for x in line.rstrip().split()]

        embeddings = np.zeros((voc_size, dimension), dtype=np.float32)

        binary_len = np.dtype(np.float32).itemsize * dimension
        count = 0
        for idx in range(voc_size):
            # mixed text and binary: read text first, then binary
            node = []
            while True:
                ch = fin.read(1)
                if ch == b" ":
                    break
                if ch == b"":
                    raise EOFError(
                        "unexpected end of input; is count incorrect or file otherwise damaged?",
                    )
                if (
                    ch != b"\n"
                ):  # ignore newlines in front of words (some binary files have)
                    node.append(ch)
            node = to_unicode(b"".join(node))

            count += 1

            word2idx[node] = len(word2idx)
            idx2word.append(node)
            embeddings[word2idx[node]] = np.fromstring(
                fin.read(binary_len),
                dtype=np.float32,
            ).astype(np.float32)

        logger.info("done")
        if count < voc_size:
            raise ValueError(
                "embeddings incomplete %s. Expecting :%d, get :%d"
                % (filename, voc_size, count),
            )
    return {
        "embeddings": embeddings,
        "word2idx": word2idx,
        "idx2word": idx2word,
    }


def read_h5_embeddings(filename, idx2word=[], word2idx={}):
    # map indexes of word vectors in matrix to their corresponding words

    reg_vectors_df = pd.read_hdf(filename)
    reg_index = reg_vectors_df.index.values
    idx2word = [""] * len(reg_index)

    for idx, w in enumerate(reg_index):
        if w in word2idx:
            continue
        word2idx[w] = idx

    for word, idx in word2idx.items():
        idx2word[idx] = word

    # convert to numpy array
    embeddings = reg_vectors_df.values

    return {
        "embeddings": embeddings,
        "word2idx": word2idx,
        "idx2word": idx2word,
    }


def read_txt_embeddings(filename, idx2word=[], word2idx={}, shape_header=True):
    # map indexes of word vectors in matrix to their corresponding words
    embeddings = []
    with open(filename, "rb") as f:
        if shape_header:
            voc_size, dimension = [int(x) for x in f.readline().rstrip().split()]
            voc_size = voc_size
            embeddings = np.zeros((voc_size, dimension), dtype=np.float32)
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                logger.info(f"Saving binary embeddings: {idx}")
            line = line.rstrip().split()
            word = line[0]
            if not word:
                continue
            word2idx[word] = len(word2idx)
            embeddings.append([float(x) for x in line[1:]])
    # convert to numpy array
    embeddings = np.array(embeddings)
    return {
        "embeddings": embeddings,
        "word2idx": word2idx,
        "idx2word": idx2word,
    }


def write_txt_embeddings(filename, embeddings, idx2word, shape_header=True):
    with open(filename, "wb") as fout:
        if shape_header:
            fout.write("%s %s\n" % embeddings.shape)
        for idx, item in enumerate(idx2word):
            if idx % 10000 == 0:
                logger.info(f"Saving binary embeddings: {idx}")
            fout.write(item + " ")
            fout.write(" ".join([str(x) for x in embeddings[idx]]))
            fout.write("\n")
