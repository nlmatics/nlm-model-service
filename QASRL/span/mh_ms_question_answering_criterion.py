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


@register_criterion("mh_ms_question_answering")
class mhmsQuestionAnsweringCriterion(FairseqCriterion):
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
            logging_output["ncorrect"] = (preds == targets).sum()

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


        
        # shape of logits:  Batch_size X sequence_length X  7 * 2 
        logits_ = logits.view( logits.shape[:-1] + (7, 2))
        loss = 0
        IO_encoded = []
        for i in range(7):
            IO_positions = sample[ f"label_{i}" ] 
            io_encoded = self.IO_encode(IO_positions, logits.size(1))  #io_en (shape): batch_size X seq_lenth 
            IO_encoded.append(io_encoded)
            #start_loss = self._compute_loss(start_logits[ :, :, i], start_positions)
            #end_loss = self._compute_loss(end_logits[ :, :, i], end_positions)

            loss += self._compute_loss(logits_[:, :, i, :], io_encoded)

        loss = loss / 7    
        sample_size = logits.size(0)
        logging_output = {
            "loss": loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        #logits_(shape): Batch_size X sequence_length X 7 X 2
        IO_encoded = torch.stack(IO_encoded, dim = -1)
        if IO_encoded is not None and logits is not None:
            correct_label = logits_.argmax(dim = -1) == IO_encoded
            logging_output["ncorrect"] = correct_label.sum() 

        return loss, sample_size, logging_output

    def _compute_loss(self, logits, IO_encoded):
        
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, 2), IO_encoded.view(-1))
        
        return loss
  
    def IO_encode(self, IO_positions, seq_length):
        """
        IO_positions shape : Batch_size x 400
        All the positions are at the begining of each row and the rest are zeros
        """
        #changing the size to seq_len
        IO_positions_ = IO_positions[:, :seq_length]
        #one_hot_encoding (shape): Batch_size X seq_length(span_positions) X seq_length
        one_hot_encoding = torch.nn.functional.one_hot(IO_positions_, seq_length).long()
        IO_encoded = one_hot_encoding.sum(dim = 1)
        IO_encoded[:, 0] = 0
        return IO_encoded.clip(0, 1)
        



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
