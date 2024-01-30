import torch
import torch.nn.functional as F
from fairseq.data.data_utils import collate_tokens
from flask_restful import inputs

from .roberta_resource import BaseRobertaResource
from .utils import punctuation


class RobertaClassificationResource(BaseRobertaResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create map between label and predicted id
        self.label2id = {}
        self.id2label = []

        for label in self.model.task.label_dictionary.symbols:
            if label.startswith(("<", "madeupword")) or label.isdigit():
                continue
            self.label2id[label] = len(self.label2id)
            self.id2label.append(label)

        self.logger.info(
            f"{self.model_name} has {len(self.id2label)} labels: {self.id2label}",
        )

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
        self.post_req_parser.add_argument(
            "keep_statement",
            type=str,
            help="just for boolq, changes question to statement if true"
        )

        # put request arguments
        self.put_req_parser.add_argument(
            "questions",
            type=str,
            action="append",
            help="List of questions",
        )
        self.put_req_parser.add_argument(
            "sentences",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.put_req_parser.add_argument(
            "labels",
            type=str,
            action="append",
            help="List of labels",
        )
        self.put_req_parser.add_argument(
            "update_workers",
            type=str,
            help="Update workers with master weights, if true",
        )
        self.put_req_parser.add_argument(
            "save_model",
            type=str,
            help="Saves the model",
        )
        self.put_req_parser.add_argument(
            "restart_workers",
            type=str,
            help="Restart workers with the latest checkpoint file unless a restart_checkpoint file is mentioned",
        )
        self.put_req_parser.add_argument(
            "restart_checkpoint",
            type=str,
            help="restart checkpoint file to be used for restarting",
        )

        # backward compatibility

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
        keep_statement = inputs.boolean(request.pop("keep_statement") or False)
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
            tokens = self.encode(questions, sentences, keep_statement=keep_statement)
        else:
            tokens = self.encode(questions or sentences, keep_statement=keep_statement)

        return tokens, request

    def parse_put_request(self):
        request = self.put_req_parser.parse_args()
        save_model = inputs.boolean(request["save_model"] or False)


        batch_size = 4

        if torch.cuda.is_available():  
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # get query
        questions = request.pop("questions") or []

        sentences = request.pop("sentences") or []

        labels = request.pop("labels") or []

        # sentence pairs
        is_sentence_pairs = questions and sentences

        # check length for input pairs
        if len(questions) != len(labels):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and labels ({questions} != {labels})",
            )
            raise ValueError("mismatched input size")
        if is_sentence_pairs and len(questions) != len(sentences):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
            )
            raise ValueError("mismatched input size")

        samples = []
        labels_ = []
        for index in range(len(questions)):

            question = questions[index] if questions else None
            sentence = sentences[index] if sentences else None

            label = labels[index]
            # check label
            if label not in self.label2id:
                raise ValueError(
                    f"undefined label, possible choices are {self.id2label}",
                )

            if is_sentence_pairs:
                tokens = self.encode(question, sentence)
            else:
                tokens = self.encode(question or sentence)

            batch_tokens = collate_tokens(
                tokens,
                pad_idx=1,
            )

            # sample for active learning
            shape = batch_tokens.shape

            samples.append(tokens[0].to(device))
            labels_.append(self.label2id[label])

        data = [{
            "tokens" : samples, 
            "target": torch.LongTensor(labels_).to(device), 
            "model": "classification",
            "batch_size": batch_size
        }]    
        
        restart_dict = {}

        return data, save_model, restart_dict

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        outputs = {}

        if kwargs.get("return_labels", True):
            predictions = [logit.argmax(dim=0).numpy().tolist() for logit in logits]
            outputs["predictions"] = []
            for p in predictions:
                if p < len(self.id2label):
                    prediction = self.id2label[p]
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
        if self.model_name != "BOOLQ":
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

    def encode(self, questions, sentences=[], keep_statement=False):
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(sentences, str):
            sentences = [sentences]

        tokens = []
        is_sentence_pair = questions and sentences

        # preprocess text
        if is_sentence_pair:
            questions = [self.preprocess_question(x) for x in questions]
            # Performance improvement for BOOLQ_MODEL
            if sentences.count(sentences[0]) == len(sentences):
                sentences = [self.preprocess_sentence(sentences[0])] * len(sentences)
            else:
                sentences = [self.preprocess_sentence(x) for x in sentences]
        else:
            questions = [self.preprocess_sentence(x) for x in questions]

        for index in range(len(questions)):
            if is_sentence_pair:
                token = self.model.encode(questions[index], sentences[index])
            else:
                token = self.model.encode(questions[index])

            tokens.append(token)
        return tokens
