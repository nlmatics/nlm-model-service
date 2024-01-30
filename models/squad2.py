import collections
import logging
import math
from timeit import default_timer

import numpy as np
import torch
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer


class BaseQA(Resource):
    def __init__(
        self,
        name,
        model,
        tokenizer,
        batch_size=8,
        max_length=512,
        doc_stride=128,
        max_query_length=64,
        use_gpu=True,
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

        self.batch_size = batch_size

        # The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated,
        # and sequences shorter than this will be padded.
        self.max_seq_length = max_length
        # When splitting up a long document into chunks, how much stride to take between chunks.
        self.doc_stride = doc_stride
        # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
        self.max_query_length = max_query_length

        self.device = "cpu"
        if use_gpu and torch.cuda.device_count():
            self.device = "cuda"
        self.model.to(self.device)

        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "question", type=str, action="append", help="Question text",
        )
        self.req_parser.add_argument("text", type=str, action="append", help="Text")
        self.req_parser.add_argument("batch_size", type=int, help="Text")

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def post(self):
        """Handler for the POST request
        :return: predictions in JSON
        """
        args = self.req_parser.parse_args()
        texts = args["text"]
        questions = args["question"]
        batch_size = args["batch_size"]

        batch_size = batch_size if batch_size else self.batch_size

        if len(texts) != len(questions):
            raise Exception("size mismatch: text and questions must be of same length")
        self.logger.debug(f"Running {self.name} inference on {len(questions)} samples")

        start = default_timer()
        answers = self.run_qa(texts, questions, batch_size)
        self.logger.debug(
            f"{len(texts)} sentences finished in {default_timer() - start}s",
        )
        return jsonify({"answers": [answers, {}]})

    def run_qa(self, contexts, questions, batch_size):

        examples = []
        for idx, (question, context) in enumerate(zip(questions, contexts)):
            example = SquadExample(
                qas_id=idx,
                question_text=question,
                context_text=context,
                answer_text="",
                start_position_character=None,
                title=None,
                answers=[],
                is_impossible=False,
            )
            examples.append(example)

        features, dataset = squad_convert_examples_to_features(
            examples,
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            return_dataset="pt",
        )

        eval_dataloader = DataLoader(dataset, batch_size=batch_size)

        all_results = []
        all_start_logits = []
        all_end_logits = []
        for batch in tqdm(eval_dataloader, desc="Infering"):
            self.model.eval()

            batch = trim_batch(batch)

            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                example_indices = batch[3]

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output

                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)
                all_start_logits.append(max(start_logits))
                all_end_logits.append(max(end_logits))

        predictions, probs = compute_predictions_logits(
            self.tokenizer, examples, features, all_results,
        )

        answers = {}
        for idx in range(len(predictions)):
            answers[idx] = {
                "probability": probs[idx],
                "start_logit": all_start_logits[idx],
                "end_logit": all_end_logits[idx],
                "text": predictions[idx],
            }
        return answers


"""Below is some help functions/classes from transformers"""


def compute_predictions_logits(
    tokenizer,
    all_examples,
    all_features,
    all_results,
    n_best_size=1,
    max_answer_length=512,
    do_lower_case=True,
    #     output_prediction_file,
    #     output_nbest_file,
    #     output_null_log_odds_file,
    verbose_logging=False,
    version_2_with_negative=True,
    null_score_diff_threshold=0,
):
    #     """Write final predictions to the json file and log-odds of null if needed."""
    #     logger.info("Writing predictions to: %s" % (output_prediction_file))
    #     logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_probs = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        ),
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                ),
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"],
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging,
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                ),
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit,
                    ),
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0, _NbestPrediction(text="", start_logit=0.0, end_logit=0.0),
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_probs[example.qas_id] = nbest_json[0]["probability"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            #             scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_probs[example.qas_id] = nbest_json[0]["probability"]

    #         all_nbest_json[example.qas_id] = nbest_json

    #     with open(output_prediction_file, "w") as writer:
    #         writer.write(json.dumps(all_predictions, indent=4) + "\n")

    #     with open(output_nbest_file, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    #     if version_2_with_negative:
    #         with open(output_null_log_odds_file, "w") as writer:
    #             writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, all_probs


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # if verbose_logging:
        #     logger.info("Unable to find text: '%s' in '%s'" %
        #                 (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        # if verbose_logging:
        #     logger.info(
        #         "Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        # if verbose_logging:
        #     logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        # if verbose_logging:
        #     logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


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


# build a SqaudExample


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    start_position_character + len(answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def squad_convert_example_to_features(
    example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training,
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            # logger.warning("Could not find answer: '%s' vs. '%s'",
            #                actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # if is_training and not example.is_impossible:
    #     tok_start_position = orig_to_tok_index[example.start_position]
    #     if example.end_position < len(example.doc_tokens) - 1:
    #         tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    #     else:
    #         tok_end_position = len(all_doc_tokens) - 1

    #     (tok_start_position, tok_end_position) = _improve_answer_span(
    #         all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
    #     )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, max_length=max_query_length,
    )
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length
            - doc_stride
            - len(truncated_query)
            - sequence_pair_added_tokens,
            truncation_strategy="only_second"
            if tokenizer.padding_side == "right"
            else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][
                    : encoded_dict["input_ids"].index(tokenizer.pad_token_id)
                ]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"])
                    - 1
                    - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][
                    last_padding_id_position + 1 :
                ]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = (
                len(truncated_query) + sequence_added_tokens + i
                if tokenizer.padding_side == "right"
                else i
            )
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = (
            len(truncated_query) + sequence_added_tokens
        )
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(
                spans, doc_span_index, doc_span_index * doc_stride + j,
            )
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"]
                + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        # if is_training and not span_is_impossible:
        #     # For training, if our document chunk does not contain an annotation
        #     # we throw it out, since there is nothing to predict.
        #     doc_start = span["start"]
        #     doc_end = span["start"] + span["length"] - 1
        #     out_of_span = False

        #     if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
        #         out_of_span = True

        #     if out_of_span:
        #         start_position = cls_index
        #         end_position = cls_index
        #         span_is_impossible = True
        #     else:
        #         if tokenizer.padding_side == "left":
        #             doc_offset = 0
        #         else:
        #             doc_offset = len(truncated_query) + sequence_added_tokens

        #         start_position = tok_start_position - doc_start + doc_offset
        #         end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                example_index=0,
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            ),
        )
    return features


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    return_dataset=False,
    threads=1,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    #     # Defining helper methods
    #     features = []
    #     threads = min(threads, cpu_count())
    #     with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
    #         annotate_ = partial(
    #             squad_convert_example_to_features,
    #             max_seq_length=max_seq_length,
    #             doc_stride=doc_stride,
    #             max_query_length=max_query_length,
    #             is_training=is_training,
    #         )
    #         features = list(
    #             tqdm(
    #                 p.imap(annotate_, examples, chunksize=32),
    #                 total=len(examples),
    #                 desc="convert squad examples to features",
    #             )
    #         )

    # build features without multithread
    features = []
    for example in examples:
        feature = squad_convert_example_to_features(
            example,
            tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
        )
        features.append(feature)

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        #         if not is_torch_available():
        #             raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long,
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long,
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor(
            [f.is_impossible for f in features], dtype=torch.float,
        )

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_example_index,
                all_cls_index,
                all_p_mask,
            )
        else:
            all_start_positions = torch.tensor(
                [f.start_position for f in features], dtype=torch.long,
            )
            all_end_positions = torch.tensor(
                [f.end_position for f in features], dtype=torch.long,
            )
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset
    # elif return_dataset == "tf":
    #     if not is_tf_available():
    #         raise RuntimeError(
    #             "TensorFlow must be installed to return a TensorFlow dataset.")

    #     def gen():
    #         for ex in features:
    #             yield (
    #                 {
    #                     "input_ids": ex.input_ids,
    #                     "attention_mask": ex.attention_mask,
    #                     "token_type_ids": ex.token_type_ids,
    #                 },
    #                 {
    #                     "start_position": ex.start_position,
    #                     "end_position": ex.end_position,
    #                     "cls_index": ex.cls_index,
    #                     "p_mask": ex.p_mask,
    #                     "is_impossible": ex.is_impossible,
    #                 },
    #             )

    #     return tf.data.Dataset.from_generator(
    #         gen,
    #         (
    #             {"input_ids": tf.int32, "attention_mask": tf.int32,
    #                 "token_type_ids": tf.int32},
    #             {
    #                 "start_position": tf.int64,
    #                 "end_position": tf.int64,
    #                 "cls_index": tf.int64,
    #                 "p_mask": tf.int32,
    #                 "is_impossible": tf.int32,
    #             },
    #         ),
    #         (
    #             {
    #                 "input_ids": tf.TensorShape([None]),
    #                 "attention_mask": tf.TensorShape([None]),
    #                 "token_type_ids": tf.TensorShape([None]),
    #             },
    #             {
    #                 "start_position": tf.TensorShape([]),
    #                 "end_position": tf.TensorShape([]),
    #                 "cls_index": tf.TensorShape([]),
    #                 "p_mask": tf.TensorShape([None]),
    #                 "is_impossible": tf.TensorShape([]),
    #             },
    #         ),
    #     )

    return features


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def trim_batch(batch):
    max_len = 0
    for b in batch[0].numpy():
        max_len = max(max_len, np.trim_zeros(b, "b").shape[0])

    _batch = (
        batch[0][:, :max_len],
        batch[1][:, :max_len],
        batch[2][:, :max_len],
        batch[3],
        batch[4],
    )

    return _batch


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(
        self,
        unique_id,
        start_logits,
        end_logits,
        start_top_index=None,
        end_top_index=None,
        cls_logits=None,
    ):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# for local testing
if __name__ == "__main__":
    from transformers import AlbertTokenizer, AlbertForQuestionAnswering

    tokenizer = AlbertTokenizer.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")

    model = AlbertForQuestionAnswering.from_pretrained(
        "ahotrod/albert_xxlargev1_squad2_512",
    )

    qa = BaseQA("albert-test", model, tokenizer, batch_size=3)

    question1 = "Who is Jim Henson?"
    answer1 = "Jim Henson is a nice puppet"

    question2 = "Who is Jan Henson?"
    answer2 = "Jan Henson is a puppeteer"

    res = qa.run_qa(
        [question1, question2, question2, question1],
        [answer1, answer2, answer2, answer1],
        3,
    )
