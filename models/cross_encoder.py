import logging

import torch
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource


class CrossEncoder(Resource):
    def __init__(
        self,
        name,
        model,
        tokenizer,
        representations=None,
        batch_size=1,
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
            "left_sentences",
            type=str,
            action="append",
            help="First list of sentences",
        )
        self.req_parser.add_argument(
            "right_sentences",
            type=str,
            action="append",
            help="Second list of sentences",
        )

    def post(self):
        """Handle post requests to run inference

        Returns:
            json: the jsonified predictions for on the provided input
        """
        print("something else!!")

        args = self.req_parser.parse_args()
        left_sents = args["left_sentences"]
        right_sents = args["right_sentences"]

        # check if is return label
        return_logits = True

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
            logits = self.predict(
                left_sents,
                right_sents,
            )
            # return prediction results

            data = {"logits": ""}
            if return_logits:
                data["logits"] = logits
            return jsonify(data)

        # throw exception
        except Exception as e:
            msg = {"status": "fail", "message": str(e), "predictions": None}
            self.logger.error(str(e))
            if self.representations:
                return msg
            else:
                return jsonify(msg)

    def predict(self, questions, contexts, batch_size=1):
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
            question + "</s>" + context[: self.max_length - 3 - len(question)]
            for question, context in zip(questions, contexts)
        ]
        logits = []

        for i in range(0, len(texts)):
            # input for current batch
            inputs = self.tokenizer.batch_encode_plus(
                texts[i : i + batch_size],
                add_special_tokens=True,
                return_tensors="pt",
                pad_to_max_length=True,
            )

            with torch.no_grad():
                # outputs
                _logits = self.model(
                    inputs.to(self.device),
                )[0]

            logits += _logits.tolist()

        return logits


# for local testing
if __name__ == "__main__":
    from transformers import AutoTokenizer, RobertaModel
    from model import RetModel

    roberta = RobertaModel.from_pretrained("roberta-base")
    model = RetModel(roberta)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    state_dict = torch.load(
        "/home/nima.sheikholeslami/cross_encoder.pt",
        map_location=torch.device("cuda"),
    )
    model.load_state_dict(state_dict)
    qnli = CrossEncoder("albert-test", model, tokenizer, batch_size=1)

    question1 = "What percentage of farmland grows wheat?"
    answer1 = (
        "More than 50% of this area is sown for wheat, 33% for barley and 7% for oats."
    )

    question2 = "Where did the Exposition take place?"
    answer2 = "This World's Fair devoted a building to electrical exhibits."

    res = qnli.predict(
        [question1, question2, question2, question1],
        [answer1, answer1, answer2, answer2],
    )
    print(res)
    # res = qnli.predict(
    #    [question1, question2, question2, question1],
    #    [answer1, answer2, answer2, answer1],
    # )
