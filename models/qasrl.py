import logging

import torch
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource


class QASRL(Resource):
    def __init__(
        self,
        spacy_nlp,
        model,
        use_gpu=True,
    ):
        """Serve huggingface.transformers models for classification tasks

        This model should be able to handle all classification tasks using
        transformer models

        Args:
            name: a str of model's name
            model: a instance of the model to serve
            tokenizer: the corresponding tokenizer of the model
            batch_size: batch_size of the model

        Returns:
            A instance of the flask resource
        """
        super().__init__()
        self.nlp = spacy_nlp

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.model = model

        # self.device = "cpu"
        # if use_gpu and torch.cuda.device_count():
        #    self.device = "cuda"
        # self.model.to(self.device)

        # argument parser
        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "sentences",
            type=str,
            help="sentences",
        )

    def post(self):
        """Handle post requests to run inference

        Returns:
            json: the jsonified predictions for on the provided input
        """

        args = self.req_parser.parse_args()
        sentences = args["sentences"]

        data = self.predict(sentences)
        # return jsonify(data)
        try:
            data = self.predict(sentences)
            return jsonify(data)
        # throw exception
        except Exception as e:
            msg = {"status": "fail", "message": str(e), "predictions": None}
            self.logger.error(str(e))
            if self.representations:
                return msg
            else:
                return jsonify(msg)

    def predict_for_verb(self, sentences, verb):
        """Run the predictions on the document and a verb

        This is a wrapper to call models to perform prediction

        Args:

        Returns:
            predictions: the predictions of the input
        """
        res = dict()
        res.update({"verb": verb})
        WH = ["where", "what", "why", "how", "when", "how much", "who"]
        tokens = self.model.encode(sentences, verb)
        with torch.no_grad():
            ans = self.model.predict("span", tokens)

        ans_ = ans.view(ans.shape[:-1] + (7, 2))
        mask = ans_.argmax(dim=-1)
        for i, wh in enumerate(WH):
            if (mask[0, :, i] == 0).all():
                continue
            else:
                res.update({wh: self.model.decode(tokens[mask[0, :, i].bool()])})
        return res

    def predict(self, sentences):

        ANS = []
        verbs = []

        doc = self.nlp(sentences)

        for token in doc:
            if "VB" in token.tag_:
                verbs.append(token.text)

        for verb in verbs:
            ANS.append(
                self.predict_for_verb(sentences, verb),
            )
        return ANS
