import logging
import math

import numpy as np
import torch
from flask_jsonpify import jsonify
from flask_restful import inputs
from flask_restful import reqparse
from flask_restful import Resource


class BaseQNLI(Resource):
    def __init__(
        self,
        name,
        model,
        tokenizer,
        representations=None,
        batch_size=96,
        max_length=512,
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
        self.representations = representations
        super().__init__()

        self.name = name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = "cpu"
        if use_gpu and torch.cuda.device_count():
            self.device = "cuda"
        self.model.to(self.device)

        # argument parser
        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "left_sentences", type=str, action="append", help="First list of sentences",
        )
        self.req_parser.add_argument(
            "right_sentences",
            type=str,
            action="append",
            help="Second list of sentences",
        )
        self.req_parser.add_argument(
            "batch_size", type=int, action="append", help="Second list of sentences",
        )
        self.req_parser.add_argument(
            "return_logits", type=str, help="Returns logits, if true",
        )

    def post(self):
        """Handle post requests to run inference

        Returns:
            json: the jsonified predictions for on the provided input
        """

        args = self.req_parser.parse_args()
        left_sents = args["left_sentences"]
        right_sents = args["right_sentences"]

        batch_size = args["batch_size"]
        # check if is return label
        return_logits = inputs.boolean(args["return_logits"])

        # check input shape
        if len(left_sents) != len(right_sents):
            self.logger.error(
                f"error while running {self.name} inference, mismatch"
                f" between number of left and right sentences "
                f"({len(left_sents)} != {len(right_sents)})",
            )
            raise Exception("mismatch of number of sentences")

        self.logger.debug(f"Running {self.name} inference on {len(left_sents)} samples")

        try:
            predictions, logits = self.predict(
                left_sents, right_sents, batch_size, return_logits=return_logits,
            )
            # return prediction results
            if self.representations:
                data = {"predictions": predictions.tostring()}
                if return_logits:
                    data["logits"] = logits.tostring()
                return data
            else:
                data = {"predictions": predictions.tolist()}
                if return_logits:
                    data["logits"] = logits.tolist()
                return jsonify(data)
        # throw exception
        except Exception as e:
            msg = {"status": "fail", "message": str(e), "predictions": None}
            self.logger.error(str(e))
            if self.representations:
                return msg
            else:
                return jsonify(msg)

    def predict(self, questions, contexts, batch_size=None, return_logits=False):
        """Run the predictions on the given pair of sentences

        This is a wrapper to call models to perform prediction

        Args:
            sent_1: array of sentences
            sent_2: array of second set of sentences
            return_label: True, if labels must be returned, else return
            predictions. default to True
            return_logits: True, if return logits, defualt to False

        Returns:
            predictions: the predictions of the input
        """

        batch_size = batch_size if batch_size else self.batch_size

        # Map to tuple (question, context)
        texts = [
            (question, context[: self.max_length - 3 - len(question)])
            for question, context in zip(questions, contexts)
        ]

        logits = []

        for i in range(0, len(texts), batch_size):
            # input for current batch
            inputs = self.tokenizer.batch_encode_plus(
                texts[i : i + batch_size],
                add_special_tokens=True,
                return_tensors="pt",
                pad_to_max_length=True,
            )
            with torch.no_grad():
                # outputs
                _logits = (
                    self.model(
                        input_ids=inputs["input_ids"].to(self.device),
                        token_type_ids=inputs["token_type_ids"].to(self.device),
                        attention_mask=inputs["attention_mask"].to(self.device),
                    )[0]
                    .cpu()
                    .numpy()
                )

            logits += _logits.tolist()
        preds = np.argmax(logits, axis=1)
        if return_logits:
            logits = np.array([_compute_softmax(x) for x in logits])
            return preds, logits
        else:
            return preds, None


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


# for local testing
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("/Users/asxzy/nlm/QNLI-albert-xxlarge-v1")

    model = AutoModelForSequenceClassification.from_pretrained(
        "/Users/asxzy/nlm/QNLI-albert-xxlarge-v1",
    )

    qnli = BaseQNLI("albert-test", model, tokenizer, batch_size=3)

    question1 = "What percentage of farmland grows wheat?"
    answer1 = (
        "More than 50% of this area is sown for wheat, 33% for barley and 7% for oats."
    )

    question2 = "Where did the Exposition take place?"
    answer2 = "This World's Fair devoted a building to electrical exhibits."

    res = qnli.predict(
        [question1, question2, question2, question1],
        [answer1, answer2, answer2, answer1],
        return_logits=False,
    )

    res = qnli.predict(
        [question1, question2, question2, question1],
        [answer1, answer2, answer2, answer1],
        return_logits=True,
    )
