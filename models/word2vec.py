import logging

import numpy as np
import spacy
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource


class Word2Vec(Resource):
    def __init__(self, *args):
        spacy.prefer_gpu()
        self._spacy_model = args[0]
        self._w2v = args[1]
        self.parser = reqparse.RequestParser()
        self.parser.add_argument(
            "left_sents", type=str, action="append", help="First set of sentences",
        )
        self.parser.add_argument(
            "right_sents", type=str, action="append", help="Second set of sentences",
        )

    def post(self):
        args = self.parser.parse_args(strict=True)
        logging.info(
            f"Left sentences: {args['left_sents']}, Right sentences:{args['right_sents']}",
        )
        if len(args["left_sents"]) == len(args["right_sents"]):
            left_sent_docs = [self._spacy_model(sent) for sent in args["left_sents"]]
            right_sent_docs = [self._spacy_model(sent) for sent in args["right_sents"]]
            left_vectors = []
            right_vectors = []
            for doc in left_sent_docs:
                vect = [
                    self._w2v[t.text] if t.text in self._w2v else self._w2v["unknown"]
                    for t in doc
                    if t.is_alpha
                    and not t.is_punct
                    and not t.is_digit
                    and not t.is_stop
                ]
                left_vectors.append(vect)

            for doc in right_sent_docs:
                vect = [
                    self._w2v[t.text] if t.text in self._w2v else self._w2v["unknown"]
                    for t in doc
                    if t.is_alpha
                    and not t.is_punct
                    and not t.is_digit
                    and not t.is_stop
                ]
                right_vectors.append(vect)

            # for doc in right_sent_docs
            # left_vecs = [self._wv[t.text] if t.text in self._wv else self._wv['unknown'] for t in doc for doc in left_sent_docs if t.is_alpha and not t.is_stop and not t.is_punct and not t.is_digit]
            # right_vecs = [self._wv[t.text] if t.text in self._wv else self._wv['unknown'] for t in doc for doc in right_sent_docs if t.is_alpha and not t.is_stop and not t.is_punct and not t.is_digit]
            # left_vectors = [self._w2v[token.text] if token.text in self._w2v else self._w2v['unknown'] for token in doc for doc in left_sent_docs if token.is_alpha and not token.is_stop and not token.is_punct and not token.is_digit]
            # right_vectors = [self._w2v[token.text] if token.text in self._w2v else self._w2v['unknown'] for token in doc for doc in right_sent_docs if token.is_alpha and not token.is_stop and not token.is_punct and not token.is_digit]
            return jsonify(
                {
                    "left": np.array([left_vectors]).tolist(),
                    "right": np.array([right_vectors]).tolist(),
                },
            )
        else:
            return "fail: left and right sentences are of different lengths"
