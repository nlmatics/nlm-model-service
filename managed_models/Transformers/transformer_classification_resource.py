import torch
import torch.nn.functional as F
from fairseq.data.data_utils import collate_tokens
from flask_restful import inputs

from .transformer_resource import BaseTransformersResource
from .utils import punctuation


# it may be passed through cross_encoder source code


class TransformersClassificationResource(BaseTransformersResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # post request arguments
        self.post_req_parser.add_argument(
            "questions",
            type=str,
            action="append",
            help="List of questions",
        )
        self.post_req_parser.add_argument(
            "sentences",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.post_req_parser.add_argument(
            "return_labels",
            type=str,
            help="Returns predicted label, if true, otherwise just the prediction",
        )
        self.post_req_parser.add_argument(
            "return_logits",
            type=str,
            help="Returns logits, if true",
        )
        self.post_req_parser.add_argument(
            "return_probs",
            type=str,
            help="Returns probabilities, if true",
        )

        # put request arguments
        self.put_req_parser.add_argument(
            "questions",
            type=str,
            action="append",
            help="List of questions",
        )
        self.put_req_parser.add_argument(
            "pos_sentences",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.put_req_parser.add_argument(
            "neg_sentences_1",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.put_req_parser.add_argument(
            "neg_sentences_2",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.put_req_parser.add_argument(
            "update_workers",
            type=str,
            help="Update workers with master weights, if true",
        )

        # backword compabilitable

        # post request arguments
        self.post_req_parser.add_argument(
            "left_sentences",
            type=str,
            action="append",
            help="First list of sentences",
        )
        self.post_req_parser.add_argument(
            "right_sentences",
            type=str,
            action="append",
            help="Second list of sentences",
        )

    def parse_post_request(self):
        request = self.post_req_parser.parse_args()

        # get query
        questions = request.pop("questions") or []

        sentences = request.pop("sentences") or []

        # backword compatible
        left_sentences = request.pop("left_sentences") or []
        right_sentences = request.pop("right_sentences") or []

        if not questions and not sentences:
            questions = left_sentences
            sentences = right_sentences

        request["return_labels"] = inputs.boolean(request["return_labels"] or True)
        request["return_logits"] = inputs.boolean(request["return_logits"] or False)
        request["return_probs"] = inputs.boolean(request["return_probs"] or False)

        # sentence pairs
        is_sentence_pairs = questions and sentences

        # empty query
        if not questions and not sentences:
            return [], request

        # check length for input pairs
        if is_sentence_pairs and len(questions) != len(sentences):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
            )
            raise ValueError("mismatched input size")

        if is_sentence_pairs:
            tokens = self.encode(questions, sentences)
        else:
            tokens = self.encode(questions or sentences)

        return tokens, request

    def parse_put_request(self):
        request = self.put_req_parser.parse_args()
        update_workers = inputs.boolean(request["update_workers"] or False)

        # get query
        """
        questions = request.pop("questions") or []

        pos_sentences = request.pop("pos_sentences") or []
        neg_sentences_1 = request.pop("neg_sentences_1") or []
        neg_sentences_2 = request.pop("neg_sentences_2") or []
        """
        samples = request.pop("samples") or []

        questions = []
        pos_sentences = []
        neg_sentences = []

        for sample in samples:
            questions.append(sample["question"])
            pos_sentences.append(sample["pos_sentence"])
            neg_sentences.append(sample["neg_sentence"])

        # labels = request.pop("labels") or []

        # empty query
        if not update_workers and (not questions and not pos_sentences):
            raise ValueError("require questions and/or sentences")

        # sentence pairs
        # is_sentence_pairs = questions and pos_sentences

        # check length for input pairs
        if (len(questions) != len(pos_sentences)) or (
            len(questions) != len(neg_sentences)
        ):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and labels ({questions} != {neg_sentences})",
            )
            raise ValueError("mismatched input size")

        samples = []
        # each question comes with a positive sample as the first one, and two negative samples
        for index in range(len(questions)):

            question = questions[index]
            pos_sentence = pos_sentences[index]
            neg_sentences_ = neg_sentences[index]

            # label = labels[index]
            # check label
            # if label not in self.model_manager.label2id:
            #    raise ValueError(
            #        f"undefined label, possible choices are {self.model_manager.id2label}",
            #    )

            tokens_1 = self.encode(question, pos_sentence)
            tokens_neg_sentences = [
                self.ecndoe(question, neg_sentence) for neg_sentence in neg_sentences_
            ]

            batch = [tokens_1]
            batch.extend(tokens_neg_sentences)

            batch_tokens = collate_tokens(
                batch,
                pad_idx=1,
            )
            mask = torch.ones(batch_tokens.shape)
            mask[torch.where(batch_tokens == 1)] = 0

            label = torch.zeros(1).long()
            # sample for active learning
            sample = {
                "input": {
                    "input_ids": batch_tokens,
                    "attention_mask": mask,
                },
                "target": label,
                # "start_positions": torch.LongTensor([start_token_idx]),
                # "end_positions": torch.LongTensor([end_token_idx]),
            }
            samples.append(sample)

        return samples, update_workers

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        outputs = {}

        if kwargs.get("return_labels", True):
            predictions = [logit.argmax(dim=0).numpy().tolist() for logit in logits]
            outputs["predictions"] = []
            for p in predictions:
                if p < len(self.model_manager.id2label):
                    prediction = self.model_manager.id2label[p]
                else:
                    prediction = None
                outputs["predictions"].append(prediction)

        if kwargs.get("return_probs", False):
            outputs["probs"] = [
                F.softmax(logit.float(), dim=0).numpy().tolist() for logit in logits
            ]

        if kwargs.get("return_logits", False):
            outputs["logits"] = [logit.numpy().tolist() for logit in logits]

        return outputs

    def preprocess_question(self, text):
        text = text.strip()
        if not text:
            return ""
        # uppercase first letter
        text = text[0].upper() + text[1:]
        if not text.endswith("?"):
            text += "?"
        return text

    def preprocess_sentence(self, text):
        text = text.strip()
        if not text:
            return ""
        # uppercase first letter
        text = text[0].upper() + text[1:]
        if not text.endswith(punctuation):
            text += "."
        return text

    def encode(self, questions, sentences=[]):
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(sentences, str):
            sentences = [sentences]

        tokens = []
        is_sentence_pair = questions and sentences

        # preprocess text
        if is_sentence_pair:
            questions = [self.preprocess_question(x) for x in questions]
            sentences = [self.preprocess_sentence(x) for x in sentences]
        else:
            questions = [self.preprocess_sentence(x) for x in questions]

        for index in range(len(questions)):
            if is_sentence_pair:

                token = self.model_manager.encode(
                    questions[index] + "</s>" + sentences[index],
                    return_tensors="pt",
                ).input_ids.squeeze()
            else:
                token = self.model_manager.encode(
                    questions[index],
                    return_tensors="pt",
                ).input_ids.squeeze()

            tokens.append(token)
        return tokens
