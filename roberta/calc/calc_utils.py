import logging
import os
import re
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data.data_utils import collate_tokens
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource
from scipy.special import softmax

from models.parallel_utils import predict_smart_batch
from roberta.calc.calc_criterion import CriterionTask
from roberta.calc.calc_model import CalculationModel
from roberta.calc.calc_task import CalculationTask

# flake8: noqa


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_calc_model(
    path,
    checkpoint="model.pt",
    gpt2_encoder_json=None,
    gpt2_vocab_bpe=None,
    **kwargs,
):
    model = CalculationModel.from_pretrained(
        path,
        checkpoint_file=checkpoint,
        arch="calc_roberta_span_large",
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    # model_name = "roberta_span"
    # model = model.half()
    # if torch.cuda.is_available():
    #     logger.info(f"Using GPU for {model_name}")
    #     model.cuda()

    # model.eval()
    return model
