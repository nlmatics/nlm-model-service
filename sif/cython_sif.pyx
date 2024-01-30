# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import cython
cimport numpy as np
import numpy as np

from libcpp.string cimport string
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector
from .embedding_utils import read_binary_embeddings, read_txt_embeddings

import pandas as pd
from nltk.tokenize import WordPunctTokenizer

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ctypedef np.ndarray ndarray
ctypedef np.float32_t REAL_t
ctypedef np.uint32_t UINT_t
REAL = np.float32
INT = np.int32

import time

from sklearn.preprocessing import normalize



cdef UINT_t RAND_MAX = 2**32 - 1
cdef REAL_t HALF = 0.5
cdef char CHAR_A = b"A"
cdef char CHAR_z = b"z"
cdef char CHAR_1 = b"1"

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline UINT_t random_int32(
    unsigned long long *next_random
) nogil:
    cdef UINT_t this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random


cdef inline REAL_t random(
    unsigned long long *next_random
) nogil:
    return <REAL_t>random_int32(next_random)/<REAL_t>RAND_MAX

cdef inline unsigned long long dummy_hash(
    string s,
    unsigned long long vocab_size
) nogil:
    cdef UINT_t h = 0
    cdef unsigned long long hash_value
    cdef unsigned long long index = 1
    cdef char b
    for b in s:
        hash_value = <unsigned long long>b
        random_int32(&hash_value)
        hash_value = hash_value * index
        h += hash_value
        index += 1

    # prevent overlapping
    if h < vocab_size:
        h += vocab_size
    return h



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef class SIFModel:
    # pointer of the word embeddings
    cdef ndarray word_embeddings
    cdef REAL_t rmpc
    cdef cpp_map[string, unsigned long long] word2idx
    cdef cpp_map[string, REAL_t] wordfreq
#     cdef vector[REAL_t] weight4ind
    cdef REAL_t weightpara
    cdef unsigned long long dimension
    cdef unsigned long long vocab_size

    def __cinit__(
        SIFModel self,
        string embedding_file,
        string global_weightfile,
        REAL_t weightpara=1e-8,
        REAL_t rmpc=0,
        do_normalize=False,
    ):
        self.rmpc = rmpc
        self.weightpara = weightpara

        self.load_embeddings(embedding_file, global_weightfile, do_normalize)
        logger.info('SIF model initilized')

    def load_embeddings(
        SIFModel self,
        string embedding_file,
        string global_weightfile,
        do_normalize=False,
    ):

        # load word vectors
        word2idx = {}
        cdef unsigned long long idx
        cdef string word


        logger.info('loading embeddings')
        # load from txt file
        if embedding_file.endswith(".bin"):
            logger.info(f"loading binary embeddings from {embedding_file}")
            res = read_binary_embeddings(embedding_file)
            embs = res['embeddings']
            word2idx = res['word2idx']
            idx2word = res['idx2word']
        elif embedding_file.endswith("txt"):
            logger.info(f"loading txt embeddings from {embedding_file}")
            embs = []
            with open(embedding_file, 'rb') as f:
                for line in f:
                    line = line.rstrip().split()
                    if len(line) != 301:
                        continue
                    word = line[0]
                    word2idx[word] = len(word2idx)
                    embs.append([float(x) for x in line[1:]])
            # convert to numpy array
            embs = np.array(embs)
        # loading from h5
        elif embedding_file.endswith("h5"):
            logger.info(f"loading h5 embeddings from {embedding_file}")
            reg_vectors_df = pd.read_hdf(embedding_file)
            reg_index = reg_vectors_df.index.values
            word2idx = {}
            for idx, w in enumerate(reg_index):
                word2idx[f"{w}".encode('utf-8', errors='ignore')] = idx
            # convert to numpy array
            embs = reg_vectors_df.values

        # dirty way to convert python dict to cpp map
        self.word2idx = word2idx
        logger.info(f"{len(word2idx)} words loaded")
        logger.info(f"{len(embs)} embeddings loaded")

        if do_normalize:
            embs = normalize(embs)

        # save np memory to word_embeddings with c memory order
        self.word_embeddings = embs.astype(REAL, order='C')
        self.vocab_size, self.dimension = embs.shape



        # load word weights
        logger.info('loading word frequency')
        # total weight
        cdef REAL_t N = 0
        if self.weightpara <= 0: # when the parameter makes no sense, use unweighted
            self.weightpara = 1.0

        global_word2weight = {}
        with open(global_weightfile, 'rb') as f:
            lines = f.readlines()
            for i in lines:
                i=i.strip()
                if(len(i) > 0):
                    i=i.split()
                    if(len(i) == 2):
                        global_word2weight[i[0]] = float(i[1])
                        N += float(i[1])
                    else:
                        logger.error(f"error when parsing line {i}")

        logger.info('align word frequency with embedding idx')

        cdef string key
        cdef REAL_t min_weight = 1
        cdef REAL_t weight
        word2weight = {}
        for key, freq in global_word2weight.items():
            weight = self.weightpara / (self.weightpara + (freq/N)**2)
            self.wordfreq[key] = weight
            word2weight[key] = weight
            min_weight = min(min_weight, word2weight[key])

        logger.info('finalizing')

    def encode(SIFModel self, sentences):
        cdef ndarray emb

        # initlize return embeddings
        emb = np.zeros((len(sentences), self.dimension), dtype=REAL)
        self._encode(sentences, <REAL_t *>emb.data)

        return emb

    cdef void _encode(
        SIFModel self,
        vector[string] sentences,
        REAL_t *emb,
    ) nogil:

        cdef string text
        cdef unsigned long long sidx = 0


        for text in sentences:
            self._encode_one_sentence(
                text,
                sidx,
                emb,
                <REAL_t *>self.word_embeddings.data,
            )
            sidx += 1

    cdef void _encode_one_sentence(
        self,
        string text,
        unsigned long long sidx,
        REAL_t *emb,
        REAL_t *word_embeddings,
    ) nogil:
        cdef unsigned long long idx
        cdef unsigned long long i
        cdef REAL_t weight
        cdef REAL_t length
        cdef string token
        cdef string token_str
        cdef vector[string] tokens
        cdef unsigned long long random_seed

        with gil:
            tokenzier = WordPunctTokenizer().tokenize
            tokens = tokenzier(text.lower())

        length = tokens.size()
        for token_str in tokens:
            token = token_str

            # get token idx
            idx = self._get_word_idx(token)
            weight = self._get_word_weight(token)
            # use idx as random_seed to generate dummy embeddings
            random_seed = idx

            if idx >= self.vocab_size:
                for i in range(self.dimension):
                    # dummy embedding is in range of [-0.5, 0.5] / dimension. see initilize in word2vec
                    emb[sidx * self.dimension + i] += (random(&random_seed) - HALF) * weight / length
            else:
                for i in range(self.dimension):
                    emb[sidx * self.dimension + i] += word_embeddings[idx * self.dimension + i] * weight / length

    cdef unsigned long long _get_word_idx(
            SIFModel self,
            string w
    ) nogil:
        cdef unsigned long long idx

        if self.word2idx.count(w):
            idx = self.word2idx[w]
        else:
            # give unique idx for ovv word
            idx = dummy_hash(w, self.vocab_size)
        return idx


    cdef REAL_t _get_word_weight(
                SIFModel self,
                string w
        ) nogil:
        cdef REAL_t weight
        cdef char b

        if self.wordfreq.count(w):
            return self.wordfreq[w]
        else:
            # give low weight for the oov
            # if is alpha, return 1
            for b in w:
                if b != CHAR_1:
                    return 1.0
            # if token is dummy_number == 1
            return 0.1
