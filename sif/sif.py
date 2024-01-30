import logging

from sif import data_io
from sif import SIF_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SIFModel:
    def __init__(
        self, wordfile, global_weightfile, local_weightfile="", weightpara=1e-8, rmpc=0,
    ):
        self.rmpc = rmpc

        # load word vectors
        self.words, self.We = data_io.getWordmap(wordfile)
        # load word weights
        # word2weight['str'] is the weight for the word 'str'
        self.word2weight = data_io.getWordWeight(
            local_weightfile, global_weightfile, weightpara,
        )
        # weight4ind[i] is the weight for the i-th word
        self.weight4ind = data_io.getWeight(self.words, self.word2weight)

    def encode(
        self, sentences, rmpc=None,
    ):
        logger.info(f"encoding {len(sentences)} sentences")
        # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        x, m = data_io.sentences2idx(sentences, self.words)
        w = data_io.seq2weight(x, m, self.weight4ind)  # get word weights

        rmpc = rmpc or self.rmpc
        # set parameters
        # get SIF embedding
        # embedding[i,:] is the embedding for sentence i
        return SIF_embedding.SIF_embedding(self.We, x, w, rmpc)
