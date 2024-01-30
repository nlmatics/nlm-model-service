#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on Fairseq code
"""

import logging
from typing import Tuple

from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel as FaiseqRobertaModel
from torch import Tensor as T
from torch import nn


from dpr.models.biencoder import BiEncoder

from dpr.utils.model_utils import (
    get_model_obj,
    load_states_from_checkpoint,
)


def get_roberta_biencoder_components(file_name):
    question_encoder = RobertaEncoder.from_pretrained(file_name)
    ctx_encoder = RobertaEncoder.from_pretrained(file_name)
    biencoder = BiEncoder(question_encoder, ctx_encoder)

    return biencoder


logger = logging.getLogger(__name__)


class RobertaEncoder(nn.Module):
    def __init__(self, fairseq_roberta_hub: RobertaHubInterface):
        super(RobertaEncoder, self).__init__()
        self.fairseq_roberta = fairseq_roberta_hub

    @classmethod
    def from_pretrained(cls, pretrained_dir_path: str):
        model = FaiseqRobertaModel.from_pretrained(pretrained_dir_path)
        return cls(model)

    def encode(self, *args, **kwargs):
        return self.fairseq_roberta.encode(*args, **kwargs)

    def forward(self, input_ids: T) -> Tuple[T, ...]:
        roberta_out = self.fairseq_roberta.extract_features(input_ids)
        cls_out = roberta_out[:, 0, :]
        # return roberta_out, cls_out, None
        return cls_out

    def predict(self, tokens):
        return self.forward(tokens)

    def get_out_size(self):
        raise NotImplementedError


def get_dpr_model(path, filename):
    saved_state = load_states_from_checkpoint(f"{path}/{filename}")

    biencoder = get_roberta_biencoder_components(path)

    # context encoder

    context_encoder = biencoder.ctx_model

    # load weights from the model file
    model_to_load = get_model_obj(context_encoder)

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state)

    # question encoder

    question_encoder = biencoder.question_model

    # load weights from the model file
    model_to_load = get_model_obj(question_encoder)

    prefix_len = len("question_model.")
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("question_model.")
    }
    model_to_load.load_state_dict(question_encoder_state)

    return context_encoder, question_encoder
