import logging
import math
from timeit import default_timer

import numpy as np
import torch
from fairseq.data.data_utils import collate_tokens
from flask_jsonpify import jsonify
from flask_restful import inputs
from flask_restful import reqparse
from flask_restful import Resource

from .boolq_utils import Question2Sentence
from models.batch_utils import Batcher
from models.batch_utils import Sample
from models.parallel_utils import predict_smart_batch


class RobertaBase(Resource):
    def __init__(
        self,
        model_name,
        encoder,
        model,
        head="sentence_classification_head",
        sentence_pair=True,
        batch_size=4096,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.model_name = model_name
        self.encoder = encoder
        self.model = model
        self.head = head

        self.max_positions = 512
        self.sentence_pair = sentence_pair
        self.batch_size = batch_size

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
        self.req_parser.add_argument(
            "return_labels",
            type=str,
            help="Returns predicted label, if true, otherwise just the prediction",
        )
        self.req_parser.add_argument(
            "return_logits",
            type=str,
            help="Returns logits, if true",
        )
        self.req_parser.add_argument(
            "return_layers",
            type=str,
            help="Returns data from each layer, if true",
        )

    def label_fn(self, label):
        return self.encoder.task.label_dictionary.string(
            [label + self.encoder.task.target_dictionary.nspecial],
        )

    def preprocess(self, text, **kwargs):
        return text

    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()

        args = self.req_parser.parse_args()

        if not args["left_sentences"]:
            return jsonify(
                {
                    "predictions": [],
                    "logits": [],
                    "layers": [],
                },
            )

        # get sentences
        left_sents = self.preprocess(args["left_sentences"], left=True)

        right_sents = self.preprocess(args["right_sentences"], right=True)

        return_labels = inputs.boolean(args["return_labels"])
        return_logits = inputs.boolean(args["return_logits"])
        return_layers = False
        if args["return_layers"] and inputs.boolean(args["return_layers"]):
            return_layers = True

        if self.sentence_pair and len(left_sents) != len(right_sents):
            self.logger.error(
                "error while running %s inference, mismatch between number of left and right sentences (%d != %d)"
                % (self.model_name, len(left_sents), len(right_sents)),
            )
            raise Exception("mismatch of number of sentences")

        self.logger.info(
            f"Start inference {self.model_name} on {len(left_sents)} pairs",
        )

        samples = []
        for index in range(len(left_sents)):
            if self.sentence_pair:
                token = self.encoder.encode(left_sents[index], right_sents[index])
            else:
                token = self.encoder.encode(left_sents[index])

            token = token[: self.max_positions]
            samples.append(Sample(index, token))

        batches, sorted_index = Batcher(
            max_token_size=self.batch_size,
        ).build_smart_batches(samples)
        # Predict smart batch
        sample_index = 0

        predictions, logits, layers = self.predict(
            batches,
            return_labels=return_labels,
            return_logits=return_logits,
            return_layers=return_layers,
        )

        for prediction, logit, layer in zip(predictions, logits, layers):
            samples[sorted_index[sample_index]].prediction = prediction
            samples[sorted_index[sample_index]].logit = logit
            samples[sorted_index[sample_index]].layer = layer
            sample_index += 1

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference {self.model_name} for {len(left_sents)} samples finished in {wall_time:.2f}ms, {wall_time/len(left_sents):.2f}ms per sample",
        )
        return jsonify(
            {
                "predictions": [x.prediction for x in samples],
                "logits": [x.logit for x in samples] if return_logits else [],
                "layers": [x.encoding for x in samples] if return_layers else [],
            },
        )

    # except Exception as e:
    #     return jsonify({'status': 'fail', 'message': str(e), 'predictions': None})

    def predict(
        self,
        batches,
        return_labels=False,
        return_logits=False,
        return_layers=False,
    ):
        """Run the predictions on the given pair of sentences
        :param sent_1: array of sentences
        :param sent_2: array of second set of sentences
        :param return_label: True, if labels must be returned, else return predictions
        :param return_logits: True, if return logits
        :return: predictions
        """
        # self.logger.debug(
        #    f"Running [{self.model_name}] prediction on [{sent_1}] and [{sent_2}], sample size: {len(sent_1)}")
        try:

            gpu_time = default_timer()
            logits = predict_smart_batch(
                self.model,
                batches,
                head=self.head,
                return_logits=return_logits,
            )

            logits = np.array(logits)
            gpu_time = (default_timer() - gpu_time) * 1000
            self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")

            predictions = logits.argmax(axis=1).tolist()
            out_features = logits.shape[1]
            logits = logits.tolist()

            if out_features > 1:
                logits = [_compute_softmax(x) for x in logits]

            if return_labels:
                predictions = [self.label_fn(p) for p in predictions]

            if return_layers:
                # warning - this code does not work when model.half() is used
                # to be fixed later
                encodings = []
                for batch in batches:
                    batch_tokens = collate_tokens(
                        batch,
                        pad_idx=1,
                    )
                    layer_features = self.encoder.extract_features(
                        batch_tokens,
                        return_all_hiddens=False,
                    )
                    layer_encodings = (
                        torch.mean(layer_features[1:-1], dim=1)
                        .cpu()
                        .detach()
                        .numpy()
                        .tolist()
                    )
                    encodings.extend(layer_encodings)
                return predictions, logits, encodings
            else:
                return predictions, logits, [None for i in range(len(predictions))]

        except Exception as e:
            self.logger.error("Error while running predictions, err: %s" % e)
            raise e


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


class RobertaQNLI(RobertaBase):
    def __init__(self, encoder, model):
        super().__init__("QNLI", encoder, model)


class RobertaMNLI(RobertaBase):
    def __init__(self, encoder, model):
        super().__init__("MNLI", encoder, model, head="mnli")


class RobertaSTSB(RobertaBase):
    def __init__(self, encoder, model):
        super().__init__("STS-B", encoder, model, batch_size=1024)


class RobertaBOOLQ(RobertaBase):
    def __init__(self, encoder, model, spacy_nlp=None):
        super().__init__("BOOL-Q", encoder, model, batch_size=1024)
        self.question_to_sentence = Question2Sentence(spacy_nlp)

    def preprocess(self, texts, **kwargs):
        if kwargs.get("left", False):
            return self.question_to_sentence(texts)
        else:
            return texts


class RobertaRothWithQ(RobertaBase):
    def __init__(self, encoder, model):
        super().__init__(
            "RothWithQ",
            encoder,
            model,
            sentence_pair=False,
            batch_size=1024,
        )
