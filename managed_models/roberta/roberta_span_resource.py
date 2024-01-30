import torch
import torch.nn.functional as F
import numpy as np
from fairseq.data.data_utils import collate_tokens
from flask_restful import inputs

from .roberta_resource import BaseRobertaResource
from .utils import punctuation


class RobertaSpanResource(BaseRobertaResource):
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
            "save_model",
            type=str,
            help="save model to disk if true",
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
        
        tokens = []
        # empty query
        if questions is not None and sentences is not None:
            # check length for input pairs
            if len(questions) != len(sentences):
                self.logger.error(
                    f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
                )
                raise ValueError("mismatched input size")

            tokens = self.encode(questions, sentences)

        return tokens, request

    def cleanup_answer(self, answer):
        answer = answer.strip()
        while len(answer) > 1 and answer[-1] in [
            ",",
            ":",
            ";",
            "("
        ]:
            answer = answer[0:-1]
        if len(answer) > 1 and answer[-1] == "." and not (answer[0].isupper() or answer[0].isdigit()):
            answer = answer[0:-1]
        if answer[0] in ["("] and ")" not in answer:
            answer = answer[1:]
        return answer
    
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

     
    def find_tok_idx(self, tokens: torch.LongTensor, answer_start, answer_end ):
        # print(answer_start, answer_end)
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        bos_offset = 0
        if tokens[0] == self.model.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
            bos_offset = 1
        char_idx = 0
        non_ascii = []
        start_tok_idx = 1
        end_tok_idx = 1
        if answer_start == -1:
            return start_tok_idx, end_tok_idx
        for idx, t in enumerate(tokens):
            word = self.model.decode(torch.tensor([t]))
            if word.isascii():
                if len(non_ascii) > 0:
                    non_ascii_word_start = char_idx
                    prev_word = self.model.decode(torch.tensor(non_ascii))
                    non_ascii_word_end = char_idx + len(prev_word)
                    # print('non ascii', idx, prev_word, non_ascii_word_start, non_ascii_word_end, answer_end >= non_ascii_word_start and answer_end < non_ascii_word_end)
                    if answer_start >= non_ascii_word_start and answer_start <= non_ascii_word_end:
                        start_tok_idx = idx - len(non_ascii)
                        # print("-----non ascii start - using tok start", start_tok_idx)
                    if answer_end >= non_ascii_word_start and answer_end <= non_ascii_word_end:
                        end_tok_idx = idx - len(non_ascii)
                        # print("-----non ascii end - using tok end", end_tok_idx)
                        break
                    non_ascii = []
                    char_idx = non_ascii_word_end
                
                word_start = char_idx
                word_end = char_idx + len(word)
                # print(idx, word, word_start, word_end, answer_end >= word_start and answer_end < word_end)
                if answer_start >= word_start and answer_start <= word_end:
                    start_tok_idx = idx
                    # print("------ascii start - using tok start", start_tok_idx)

                if answer_end >= word_start and answer_end <= word_end:
                    end_tok_idx = idx
                    # print("------ascii end - using tok end", end_tok_idx)
                    break
                char_idx = word_end
            else:
                non_ascii.append(t)
        if end_tok_idx < start_tok_idx and start_tok_idx >=0:
            end_tok_idx = start_tok_idx
            
        return start_tok_idx + bos_offset, end_tok_idx + bos_offset + 1


    def parse_put_request(self):
        request = self.put_req_parser.parse_args()
        batch_size = 4
        # get query
        save_model = inputs.boolean(request["save_model"] or False)

        questions = request.pop("questions", [])

        sentences = request.pop("sentences", [])

        answers = request.pop("answers", [])

        if torch.cuda.is_available():  
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # # empty query
        # if not update_workers and (not questions or not sentences):
        #     raise ValueError("require questions and/or sentences")

        if questions is not None and sentences is not None:
            # check length for input pairs
            if len(questions) != len(sentences):
                self.logger.error(
                    f"error while running {self.model_name} inference, mismatch between number of questions and sentences ({questions} != {sentences})",
                )
                raise ValueError("mismatched input size")

            samples = []
            start_token_indices = []
            end_token_indices = []
                
            for question, sentence, answer in zip(questions, sentences, answers):

                start_char = sentence.find(answer)
                end_char = start_char + len(answer)
                start_token_idx = 1
                end_token_idx = 1
                if start_char != end_char:
                    start_token_idx, end_token_idx = self.find_tok_idx(self.model.encode(sentence), 
                                                                    start_char, 
                                                                    start_char + len(answer))                
                
                
                start_token_indices.append(start_token_idx)
                end_token_indices.append(end_token_idx)

                if (start_token_idx != -1 and start_token_idx != end_token_idx):
                    trained_answer = self.model.decode(self.model.encode(sentence)[start_token_idx:end_token_idx])
                    trained_answer_clean = self.cleanup_answer(trained_answer)
                    if trained_answer_clean != answer.strip():
                        self.logger.info("------")
                        self.logger.info(question)
                        self.logger.info("-")
                        self.logger.info(sentence)
                        self.logger.info("-")
                        self.logger.info("trained:", trained_answer_clean.strip())
                        self.logger.info("-")
                        self.logger.info("orig trained:", trained_answer)
                        self.logger.info("-")
                        self.logger.info("expected:", answer)
                        self.logger.info("------")
                    # convert to tokens
                # print(start_token_idx, end_token_idx, answer)
                tokens = self.encode(question, sentence)
                tokens[0].to(device)
                # build batch_tokens
                samples.append(tokens[0])

                # sample for active learning

            Data = {
                "tokens": samples, 
                "start_positions": torch.LongTensor(start_token_indices).to(device), 
                "end_positions": torch.LongTensor(end_token_indices).to(device),
                "batch_size": batch_size, 
                'model': 'span', 
            }

            return [Data], save_model, None
        else:
            return [], save_model, None


    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):

        # unbind logits to start positions and end positions
        start_logits = []
        start_probs = []
        start_bytes = []

        end_logits = []
        end_probs = []
        end_bytes = []

        answers = []
        probs = []

        for token, logit in zip(tokens, logits):
            start_logits, end_logits = torch.unbind(logit, dim=1)

            # get probabilities
            start_token_prob = F.softmax(start_logits.float(), dim=0)
            end_token_prob = F.softmax(end_logits.float(), dim=0)

            # get topk tokens
            _, start_topk_token = start_token_prob.topk(2, dim=0)

            _, end_topk_token = end_token_prob.topk(2, dim=0)

            # # * IMPORTANT *
            # # This is to deal with the edge case for span detection where first
            # # and second prediciton has very similar probability. e.g. 0.400 v.s. 0.399

            # If no top prediction has no span, and second prediction
            # is very close to top prediction, use second prediction
            if (
                # top prediction has no span
                end_topk_token[0] <= start_topk_token[0]
                # second prediction is very close to top prediction
                and abs(
                    end_token_prob[end_topk_token[0]]
                    - end_token_prob[end_topk_token[1]],
                )
                < 1e-3
            ):
                # swap predicted end token
                temp = end_topk_token[0]
                end_topk_token[0] = end_topk_token[1]
                end_topk_token[1] = temp

            start_prob = max(start_token_prob.numpy().tolist())
            start_probs.append(start_prob)
            start_token = start_topk_token[0]

            end_prob = max(end_token_prob.numpy().tolist())
            end_probs.append(end_prob)
            end_token = end_topk_token[0]

            prob = (start_prob + end_prob) / 2.0
            probs.append(prob)

            # extract answer text
            answer = ""
            start_byte = end_byte = -1
            # has answer
            if not (start_token == 0 or end_token == 0 or end_token <= start_token):
                answer = self.model.decode(token[start_token:end_token])
                if isinstance(answer, list):
                    # Returning list means span contains answer, override to empty string.
                    # Doing this will save us time for encoding context only
                    answer = ""
                answer = answer.strip()
            if answer:
                pre_context = self.model.decode(token[:end_token])
                end_byte = len(pre_context)
                start_byte = len(pre_context) - len(answer)
            answers.append(answer)
            start_bytes.append(start_byte)
            end_bytes.append(end_byte)

        outputs = {
            "answers": answers
        }

        if kwargs.get("return_logits", False):
            outputs["logits"] = [logit.numpy().tolist() for logit in logits]

        if kwargs.get("return_probs", False):
            outputs["start_probs"] = start_probs
            outputs["end_probs"] = end_probs
            outputs["probs"] = probs

        if kwargs.get("return_bytes", False):
            outputs["start_bytes"] = start_bytes
            outputs["end_bytes"] = end_bytes

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
            token = self.model.encode(sentences[index], questions[index])
            # deal with overflow
            if len(token) > self.model.model.max_positions():
                question_token_len = len(self.model.encode(questions[index]))
                # chunk the sentence
                token = torch.cat(
                    [
                        token[
                            : self.model.model.max_positions() - question_token_len
                        ],
                        token[question_token_len:],
                    ],
                )
            tokens.append(token)
        return tokens
