import logging
import os
import re
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource
from scipy.special import softmax

from fairseq.data.data_utils import collate_tokens
from models.parallel_utils import predict_smart_batch
from roberta.span.question_answering_criterion import QuestionAnsweringCriterion
from roberta.span.question_answering_task import QuestionAnsweringTask
from roberta.span.span_model import SpanModel

# flake8: noqa


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_span_model(
    path, checkpoint="model.pt", gpt2_encoder_json=None, gpt2_vocab_bpe=None, **kwargs
):
    model = SpanModel.from_pretrained(
        path,
        checkpoint_file=checkpoint,
        arch="roberta_span_large",
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    # model_name = "roberta_span"
    # model = model.half()
    # if torch.cuda.is_available():
    #     logger.info(f"Using GPU for {model_name}")
    #     model.cuda()

    model.cuda().eval()
    return model


# predict on a single batch of data
def batch_predict(
    model,
    batch_of_pairs,
    annotate=True,
    debug=False,
    encode=True,
    use_parallel_model=False,
    encoder=None,
):
    if encoder is None:
        encoder = model
    # encode pairs
    if encode:
        batch_tokens = []
        # batch_pair_0_tokens = []
        for pair in batch_of_pairs:
            pair_0_tokens = encoder.encode(pair[0])
            pair_1_tokens = encoder.encode(pair[1])
            if len(pair_1_tokens) > 0 and len(pair_0_tokens) > 0:
                pair_1_tokens[0] = pair_0_tokens[-1]
            pair_tokens = torch.cat((pair_0_tokens, pair_1_tokens))
            overflow = pair_tokens.size(-1) - encoder.model.max_positions()
            if overflow > 0:
                if debug:
                    logger.info(f"truncating because of overflow: {overflow}")
                pair_0_tokens = encoder.encode(pair[0])[0 : -1 * (overflow - 1)]
                pair_1_tokens = encoder.encode(pair[1])[1:]
                pair_tokens = torch.cat((pair_0_tokens, pair_1_tokens))
                batch_tokens.append(pair_tokens)
                # batch_pair_0_tokens.append(pair_0_tokens)
            else:
                batch_tokens.append(pair_tokens)
                # batch_pair_0_tokens.append(pair_0_tokens)
    # use pre calculated tokens
    else:
        batch_tokens = batch_of_pairs
        # batch_pair_0_tokens = [sample.context_token for sample in batch_of_pairs]

    if use_parallel_model:
        # change to np array for decoder
        batch = batch_tokens
    else:
        batch = collate_tokens(batch_tokens, pad_idx=1)

    # logger.debug(
    #     f"batch shape: {batch.shape}, size: {batch.shape[0]*batch.shape[1]}",
    # )

    gpu_time = default_timer()

    if use_parallel_model:
        # logits from parallel model is a list
        logits = predict_smart_batch(
            model,
            batch,
            head="span",
            return_logits=True,
        )

        logits = [np.array(x) for x in logits]

        # the parallel model shape in [smart_batch_1, smart_batch_2]
        # need to flatten to [t1, t2, t3,...]
        unflattened_batch = []
        for smart_batch in batch:
            for tokens in smart_batch:
                unflattened_batch.append(tokens)
        batch = unflattened_batch

        # calculate probs and token from the logits
        start_tokens = []
        end_tokens = []
        start_probs = []
        end_probs = []
        probs = []

        for logit in logits:
            start_logit, end_logit = logit[:, 0], logit[:, 1]

            start_prob = softmax(start_logit)
            start_token = np.argmax(start_prob)

            end_prob = softmax(end_logit)
            end_token_topk = np.argsort(end_prob)[::-1]
            if abs(end_prob[end_token_topk[0]] - end_prob[end_token_topk[1]]) < 1e-3:
                if end_token_topk[0] <= start_token:
                    end_token_topk[0], end_token_topk[1] = (
                        end_token_topk[1],
                        end_token_topk[0],
                    )
            end_token = end_token_topk[0]

            start_prob = start_prob.max()
            end_prob = end_prob.max()

            start_probs.append(start_prob)
            start_tokens.append(start_token)

            end_probs.append(end_prob)
            end_tokens.append(end_token)

            probs.append((start_prob.max() + end_prob.max()) / 2)
        probs = np.array(probs)

    else:
        logits = model.predict(
            head="span",
            tokens=batch,
            return_logits=True,
        )
        start_logits, end_logits = logits.unbind(dim=2)
        start_logits = start_logits.cpu().detach()
        end_logits = end_logits.cpu().detach()

        # start_probs, start_tokens = torch.max(F.softmax(start_logits, dim=1), dim=1)
        # end_probs, end_tokens = torch.max(F.softmax(end_logits, dim=1), dim=1)

        start_probs, start_tokens = torch.max(F.softmax(start_logits, dim=1), dim=1)
        end_probs, end_tokens = torch.topk(F.softmax(end_logits, dim=1), 2, dim=1)

        for start_token, end_token, end_prob in zip(
            start_tokens, end_tokens, end_probs
        ):
            if abs(end_prob[0] - end_prob[1]) < 1e-3:
                if end_token[0] <= start_token:
                    end_token[0], end_token[1] = end_token[1], end_token[0]

        end_probs = end_probs[:, 0]
        end_tokens = end_tokens[:, 0]

        probs = (start_probs + end_probs) / 2

    gpu_time = (default_timer() - gpu_time) * 1000
    logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")

    answers = []
    start_bytes = []
    end_bytes = []
    annotated_answers = []

    # for tokens, pair_0_tokens, start_token, end_token in zip(
    for tokens, start_token, end_token in zip(
        batch,
        # batch_pair_0_tokens,
        start_tokens,
        end_tokens,
    ):
        # if debug:
        #     logger.info("\n", c)
        #     logger.info("?: ", q)
        answer = ""
        start_byte = end_byte = -1
        if not (
            start_token == 0
            or end_token == 0
            or end_token <= start_token
            # or start_token > len(pair_0_tokens)
        ):
            answer = encoder.decode(tokens[start_token:end_token])
            pre_context = encoder.decode(tokens[:end_token])
            end_byte = len(pre_context)
            start_byte = len(pre_context) - len(answer)
            if isinstance(answer, list):
                # Returning list means span contains answer, override to empty string.
                # Doing this will save us time for encoding context only
                answer = ""
                # answer = answer[0]
            answer = answer.strip()
        answers.append(answer)
        start_bytes.append(start_byte)
        end_bytes.append(end_byte)
        if debug:
            logger.info("A: ", answer)

    if use_parallel_model:
        return (
            np.vstack(
                (start_tokens, end_tokens),
            ),
            answers,
            np.vstack(
                (start_probs, end_probs),
            ),
            probs,
            annotated_answers,
            start_bytes,
            end_bytes,
        )
    else:
        return (
            np.vstack(
                (
                    start_tokens.cpu().detach().numpy(),
                    end_tokens.cpu().detach().numpy(),
                ),
            ),
            answers,
            np.vstack(
                (start_probs.cpu().detach().numpy(), end_probs.cpu().detach().numpy()),
            ),
            probs.cpu().detach().numpy(),
            annotated_answers,
            start_bytes,
            end_bytes,
        )


def format_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"^the ", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\. .*", "", answer, flags=re.IGNORECASE)
    return answer


def create_batches(contexts, questions, bs=4):
    batches = []
    batch_of_pairs = []
    for idx, pair in enumerate(zip(contexts, questions)):
        if pair[1].endswith("?"):
            pair = (pair[0], pair[1][0:-1])
        batch_of_pairs.append(pair)
        if len(batch_of_pairs) == bs:
            batches.append(batch_of_pairs)
            batch_of_pairs = []
    if len(batch_of_pairs) > 0:
        batches.append(batch_of_pairs)
    all_pairs = [item for sublist in batches for item in sublist]
    return batches, all_pairs


def process_batches(
    model,
    batches,
    output_json=False,
    encode=True,
    encoder=None,
    use_parallel_model=False,
):
    preds = None
    probs = None
    pred_answers = []
    start_bytes = []
    end_bytes = []
    if use_parallel_model:
        (preds, pred_answers, _, probs, _, start_bytes, end_bytes,) = batch_predict(
            model,
            batches,
            encode=encode,
            encoder=encoder,
            use_parallel_model=use_parallel_model,
        )
    else:
        for idx, batch in enumerate(batches):
            try:
                (
                    batch_preds,
                    batch_answers,
                    _,
                    batch_probs,
                    _,
                    pred_start_bytes,
                    pred_end_bytes,
                ) = batch_predict(
                    model,
                    batch,
                    encode=encode,
                    encoder=encoder,
                    use_parallel_model=use_parallel_model,
                )
            except Exception as e:
                logger.error(f"error processing batch: {idx}: {e}", exc_info=True)
                bs = len(batch)
                batch_preds = np.zeros((2, bs), dtype=np.int64)
                batch_answers = [None for i in range(bs)]
                batch_probs = np.zeros((1, bs))
                pred_start_bytes = [None for i in range(bs)]
                pred_end_bytes = [None for i in range(bs)]
            if idx == 0:
                preds = batch_preds
                probs = batch_probs
            else:
                preds = np.hstack((preds, batch_preds))
                probs = np.hstack((probs, batch_probs))
            pred_answers.extend(batch_answers)
            start_bytes.extend(pred_start_bytes)
            end_bytes.extend(pred_end_bytes)

    preds = preds.T.tolist()
    probs = probs.tolist()
    if output_json:
        answers = {}
        for idx, ((start, end), answer, prob, start_byte, end_byte) in enumerate(
            zip(preds, pred_answers, probs, start_bytes, end_bytes),
        ):
            answers[idx] = {
                "probability": prob,
                "start_logit": start,
                "end_logit": end,
                "text": answer,
                "start_byte": start_byte,
                "end_byte": end_byte,
            }
        return answers
    else:
        return preds, pred_answers, probs


class Sample:
    def __init__(self, index, token, context_token):
        self.index = index
        self.token = token
        self.context_token = context_token


class RobertaSpanResource(Resource):
    def __init__(
        self,
        model_name,
        encoder,
        model,
        head="span",
        *args,
        **kwargs,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.model_name = model_name
        self.encoder = encoder
        self.model = model
        self.head = head

        self.max_token_size = 4096
        try:
            self.max_token_size = int(
                self.max_token_size * float(os.getenv("MODEL_SERVER_SCALE_FACTOR", 1)),
            )
        except Exception as e:
            self.logger.error(f"Error during parsing scaled factor: {e}")
        self.logger.info(
            f"Smart batch upperbound for {self.model_name} is {self.max_token_size} tokens",
        )

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
            "annotate",
            type=str,
            help="Annotates the answer when true",
        )
        self.req_parser.add_argument(
            "debug",
            type=str,
            help="Prints debug messages if true",
        )

    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()
        args = self.req_parser.parse_args()

        if not args["left_sentences"]:
            return jsonify({"answers": [{}, {}]})

        contexts = args["left_sentences"]
        questions = args["right_sentences"]

        if len(contexts) != len(questions):
            self.logger.error(
                "error while running %s inference, mismatch between number of left and right sentences (%d != %d)"
                % (self.model_name, len(contexts), len(questions)),
            )
            raise Exception("mismatch of number of sentences")

        self.logger.info(
            f"Roberta {self.model_name} inference on {len(contexts)} samples",
        )

        answers = self.predict(contexts, questions)

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Roberta {self.model_name} inference on {len(contexts)} samples finished in {wall_time:.2f}ms, {wall_time/len(contexts):.2f}ms per sample",
        )

        return jsonify({"answers": [answers, {}]})

    def predict(self, contexts, questions):

        # batches, all_pairs = create_batches(contexts, questions, bs=self.batch_size)

        # Build samples and maintain the orders
        samples = []
        for index in range(len(contexts)):
            # borrow the logic from batch_predict defined above
            pair_0_tokens = self.encoder.encode(contexts[index])
            pair_1_tokens = self.encoder.encode(questions[index])
            if len(pair_1_tokens) > 0 and len(pair_0_tokens) > 0:
                pair_1_tokens[0] = pair_0_tokens[-1]
            pair_tokens = torch.cat((pair_0_tokens, pair_1_tokens))
            overflow = pair_tokens.size(-1) - self.encoder.model.max_positions()
            if overflow > 0:
                pair_0_tokens = pair_0_tokens[0 : -1 * (overflow - 1)]
                # skip [CLS]
                pair_1_tokens = pair_1_tokens[1:]
                pair_tokens = torch.cat((pair_0_tokens, pair_1_tokens))

            samples.append(Sample(index, pair_tokens, pair_0_tokens))

        # Sort by token shape to build smart batch
        sorted_index = sorted(range(len(samples)), key=lambda x: samples[x].token.shape)

        # Build smart batching
        batches = []
        cur_batch = []

        # MAGIC NUMBER
        # 4096 = 512 * 8
        # This number is safe for 16GB GPU
        # Tested with other larger numbers but get slower
        for sample_index in sorted_index:
            sample = samples[sample_index]
            cur_token_size = sample.token.shape[0]
            # adding current token will cause overflow.
            # Using len(cur_batch) * cur_token_size because collate_tokens create the batch based on the longest sample
            if (len(cur_batch) + 1) * cur_token_size > self.max_token_size:
                # NOTE: since max_seq_size is 512, cur_batch will always be non-empty
                batches.append(cur_batch)
                cur_batch = []

            cur_batch.append(sample.token)
        # append reminder
        if cur_batch:
            batches.append(cur_batch)

        self.logger.info(
            f"{len(samples)} are splited into {len(batches)} smart batches",
        )

        answers = {}

        unsorted_answer = process_batches(
            self.model,
            batches,
            output_json=True,
            encode=False,
            encoder=self.encoder,
            use_parallel_model=True,
        )

        for idx, sample_index in enumerate(sorted_index):
            answers[sample_index] = unsorted_answer[idx]

        return answers


class RobertaQA(RobertaSpanResource):
    def __init__(self, encoder, model):
        super().__init__("qa", encoder, model)


class RobertaPhraseQA(RobertaSpanResource):
    def __init__(self, encoder, model):
        super().__init__("phraseqa", encoder, model)
