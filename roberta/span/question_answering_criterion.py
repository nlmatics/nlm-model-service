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


@register_criterion("question_answering")
class QuestionAnsweringCriterion(FairseqCriterion):
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
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

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
            corrects = preds == targets
            logging_output["ncorrect"] = (corrects).sum()
            logging_output["corrects"] = corrects

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="span",
        )

        start_logits, end_logits = torch.unbind(logits, dim=2)

        # shape of logits is batch_size X sequence_length X 2 (one for each position)

        start_positions = sample["start_positions"]
        end_positions = sample["end_positions"]

        start_loss = self._compute_loss(start_logits, start_positions)
        end_loss = self._compute_loss(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0

        sample_size = start_positions.size(0)
        logging_output = {
            "loss": total_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        if start_positions is not None and end_positions is not None:
            correct_start = start_logits.argmax(dim=1) == start_positions
            correct_end = end_logits.argmax(dim=1) == end_positions
            corrects = correct_start & correct_end
            logging_output["ncorrect"] = (corrects).sum()
            logging_output["corrects"] = corrects
            logging_output["starts"] = [start_logits.argmax(dim=1), start_positions]
            logging_output["ends"] = [end_logits.argmax(dim=1), end_positions]

        return total_loss, sample_size, logging_output

    def _compute_loss(self, logits, positions):
        #         print(positions)
        seq_length = logits.size(1)
        one_hot_positions = F.one_hot(positions, num_classes=600).float()
        # truncate the array to seq_length of the batch
        one_hot_positions = one_hot_positions[:, 0:seq_length]
        # lprobs = F.log_softmax(logits, dtype=torch.float32)
        # loss = F.mse_loss(lprobs, one_hot_positions, reduction='sum')
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
