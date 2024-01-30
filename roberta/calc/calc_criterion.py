# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion
from fairseq.criterions import register_criterion


@register_criterion("criterion_task")
class CriterionTask(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.args = task.args
        self.sentence_avg = sentence_avg

    def forward_sent_prediction(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head in model.classification_heads
        ), "model must provide sentence classification head for --criterion=span_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        span_logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="span",
        )

        op_logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="operation",
        )

        span_indices = sample["span_indices"]
        span_indices_ = span_indices[:, : span_logits.size(1)]
        # torch.set_printoptions(profile="full")
        # print("sample: ", sample["net_input"])
        # print("span_indices: ", span_indices)
        # print("span_indices_: ", span_indices_)
        # print("span_logits: ", span_logits)

        # span logits shape: batch_size X sequence_length X 2

        span_loss = self._compute_loss(span_logits, span_indices_)

        # op logits shape: batch_size X 3 (CLS token)

        op_label = sample["op_label"]

        op_loss = self._op_compute_loss(op_logits, op_label)

        total_loss = span_loss + op_loss

        sample_size = span_logits.size(0)

        # print("span_indicies: ", span_indices)
        # print("\n span_indices_: ", span_indices_)
        # print("\n span_logits: ", span_logits)
        # print("\n op_label: ", op_label)
        # raise

        # find logit with max prob, argmax function
        # add another for op
        # & is strong check that all is correct

        logging_output = {
            "loss": span_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        if span_logits is not None and op_logits is not None:
            correct_op = op_logits.argmax(dim=-1) == op_label
            correct_span = (span_logits.argmax(-1) == span_indices_).all(dim=-1)
            ncorrect = correct_span * correct_op
            # ncorrect = correct_span
            logging_output["ncorrect"] = ncorrect.sum(dim=0)

            logging_output.update(
                {
                    "op_accuracy": correct_op.sum(dim=0),
                    "span_accuracy": correct_span.sum(dim=0),
                },
            )

        return total_loss, sample_size, logging_output

    def _compute_loss(self, logits, positions, num_classes=600):
        #         print(positions)
        # lprobs = F.log_softmax(logits, dtype=torch.float32)
        # loss = F.mse_loss(lprobs, one_hot_positions, reduction='sum')
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, 2), positions.reshape(-1))
        return loss

    def _op_compute_loss(self, logits, operations):
        one_hot_positions = F.one_hot(operations, 5).float()
        loss = F.binary_cross_entropy_with_logits(logits, one_hot_positions)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss",
            loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss",
                loss_sum / ntokens / math.log(2),
                ntokens,
                round=3,
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy",
                100.0 * ncorrect / nsentences,
                nsentences,
                round=1,
            )

        if len(logging_outputs) > 0 and "op_accuracy" in logging_outputs[0]:
            op_accuracy = sum(log.get("op_accuracy", 0) for log in logging_outputs)
            metrics.log_scalar(
                "op_accuracy",
                100.0 * op_accuracy / nsentences,
                nsentences,
                round=1,
            )

        if len(logging_outputs) > 0 and "span_accuracy" in logging_outputs[0]:
            span_accuracy = sum(log.get("span_accuracy", 0) for log in logging_outputs)
            metrics.log_scalar(
                "span_accuracy",
                100.0 * span_accuracy / nsentences,
                nsentences,
                round=1,
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
