import torch
from fairseq.data.data_utils import collate_tokens
from flask_restful import inputs

from .roberta_resource import BaseRobertaResource
from .utils import punctuation


class RobertaCalcResource(BaseRobertaResource):
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
            "return_bytes",
            type=str,
            help="Returns answer positinos in bytes, if true",
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
            "sentences",
            type=str,
            action="append",
            help="List of sentences",
        )
        self.put_req_parser.add_argument(
            "answers",
            type=str,
            action="append",
            help="Second list of answers",
        )
        self.put_req_parser.add_argument(
            "update_workers",
            type=str,
            help="Update workers with master weights, if true",
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

        # get query
        questions = request.pop("questions", [])

        sentences = request.pop("sentences", [])

        # backword compatible
        left_sentences = request.pop("left_sentences") or []
        right_sentences = request.pop("right_sentences") or []

        request["return_bytes"] = inputs.boolean(request["return_bytes"] or False)
        request["return_logits"] = inputs.boolean(request["return_logits"] or False)
        request["return_probs"] = inputs.boolean(request["return_probs"] or False)

        if not questions and not sentences:
            # nlm_util.client sent questions as right_sentences
            questions = right_sentences
            sentences = left_sentences
            request["return_probs"] = True

        # empty query
        if not questions or not sentences:
            return []

        # check length for input pairs
        if len(questions) != len(sentences):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
            )
            raise ValueError("mismatched input size")

        tokens = self.encode(questions, sentences)

        return tokens, request

    def post(self, *args, **kwargs):
        outputs = super().post(*args, **kwargs)

        request = self.post_req_parser.parse_args()

        # get query
        if request["questions"]:
            return outputs

        old_outputs = {"answers": [{}, {}]}
        index = 0
        for key, values in outputs.items():
            for index, value in enumerate(values):
                if str(index) not in old_outputs["answers"][0]:
                    old_outputs["answers"][0][str(index)] = {}

                if key == "probs":
                    old_outputs["answers"][0][str(index)]["probability"] = value
                elif key == "answers":
                    old_outputs["answers"][0][str(index)]["text"] = value
                else:
                    old_outputs["answers"][0][str(index)][key] = value

        return old_outputs

    def parse_put_request(self):
        request = self.put_req_parser.parse_args()

        # get query
        update_workers = inputs.boolean(request["update_workers"] or False)
        restart_workers = inputs.boolean(request["restart_workers"] or False)
        restart_checkpoint = request.pop("restart_checkpoint") or ""

        questions = request.pop("questions", [])

        sentences = request.pop("sentences", [])

        answers = request.pop("answers", [])

        # empty query
        if not update_workers and (not questions or not sentences) and not restart_workers:
            raise ValueError("require questions and/or sentences")

        # check length for input pairs
        if len(questions) != len(sentences):
            self.logger.error(
                f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
            )
            raise ValueError("mismatched input size")

        samples = []
        for question, sentence, answer in zip(questions, sentences, answers):
            # get answer tokens without <s> </s>
            answer_tokens = self.model_manager.encode(answer)[1:-1]

            # find char of answer
            start_char = sentence.find(answer)
            # find start_token_idx without </s>
            # NOTE: it will find the first occured answer span
            start_token_idx = (
                len(self.model_manager.encode(sentence[:start_char].strip())) - 1
            )
            # find end_token_idx
            end_token_idx = start_token_idx + len(answer_tokens)

            # convert to tokens
            tokens = self.encode(question, sentence)
            # build batch_tokens
            batch_tokens = collate_tokens(
                tokens,
                pad_idx=1,
            )

            # sample for active learning
            sample = {
                "net_input": {
                    "src_tokens": batch_tokens,
                },
                "start_positions": torch.LongTensor([start_token_idx]),
                "end_positions": torch.LongTensor([end_token_idx]),
                "ntokens": 1,
            }
            samples.append(sample)

        restart_dict = {}
        if restart_workers:
            restart_dict["do_restart"] = True
            restart_dict["checkpoint_file"] = restart_checkpoint

        return samples, update_workers, restart_dict

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        def get_answer_index(span_io):
            spans = {"start": [], "end": []}
            n_spans = 0
            # take the bit diff
            for i, (x, y) in enumerate(zip(span_io, span_io[1:] + [0])):
                if y - x == 1:
                    spans["start"].append(i + 1)
                elif y - x == -1:
                    spans["end"].append(i + 1)
                    n_spans += 1

            return spans, n_spans

        # op_pos = ["+", "-", "*", "/"]
        answers = []

        for (span_logit, opt_logit), _tokens in zip(logits, tokens):

            span_io = span_logit.argmax(dim=-1).numpy().tolist()

            op = opt_logit.argmax(dim=0)

            spans, n_spans = get_answer_index(span_io)
            if n_spans == 0:
                answers.append("")
            else:
                _answers = []
                for start, end in zip(spans["start"], spans["end"]):
                    answer = self.model_manager.decode(_tokens[start:end])
                    _answers.append(answer.strip())

                # operations
                if n_spans == 2:
                    try:
                        if op == 0:
                            answer = float(_answers[0]) + float(_answers[1])
                        elif op == 1:
                            answer = float(_answers[0]) - float(_answers[1])
                            # answer_2 = float(_answers[1]) - float(_answers[0])
                        elif op == 2:
                            answer = float(_answers[0]) * float(_answers[1])
                        else:
                            answer = float(_answers[0]) / float(_answers[1])
                            # answer_2 = float(_answers[1]) / float(_answers[0])
                        answers.append(str(answer).strip())
                    except Exception as e:
                        self.logger.error(
                            f"Exception during eval operations, {e}",
                            exc_info=True,
                        )
                        answers.append(" ".join(_answers).strip())

                    # answers.append(answer.strip())
                else:
                    answers.append(" ".join(_answers).strip())

        outputs = {}
        outputs["answers"] = answers
        # outputs["answers_1"] = answers_1
        # outputs["answers_2"] = answers_2

        if kwargs.get("return_logits", False):
            outputs["logits"] = [
                (span_logit.numpy().tolist(), opt_logit.numpy().tolist())
                for span_logit, opt_logit in logits
            ]

        # if kwargs.get("return_probs", False):
        #     outputs["start_probs"] = start_probs
        #     outputs["end_probs"] = end_probs
        #     outputs["probs"] = probs

        # if kwargs.get("return_bytes", False):
        #     outputs["start_bytes"] = start_bytes
        #     outputs["end_bytes"] = end_bytes

        return outputs

    def preprocess_question(self, text):
        text = text.strip().lower()
        if text.endswith("?"):
            text = text[:-1]
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

    def encode(self, questions, sentences):
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(sentences, str):
            sentences = [sentences]

        # preprocess text
        questions = [self.preprocess_question(x) for x in questions]
        sentences = [self.preprocess_sentence(x) for x in sentences]

        tokens = []
        for index in range(len(questions) or len(sentences)):
            token = self.model_manager.encode(sentences[index], questions[index])
            # deal with overflow
            if len(token) > self.model_manager.max_positions():
                question_token_len = len(self.model_manager.encode(questions[index]))
                # chunk the sentence
                token = torch.cat(
                    [
                        token[
                            : self.model_manager.max_positions() - question_token_len
                        ],
                        token[question_token_len:],
                    ],
                )
            tokens.append(token)
        return tokens
