# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch.nn as nn
from fairseq.models import register_model
from fairseq.models import register_model_architecture
from fairseq.models.roberta import model as roberta_model
from fairseq.models.roberta import RobertaEncoder
from fairseq.models.roberta import RobertaHubInterface
from fairseq.models.roberta import RobertaModel

logger = logging.getLogger(__name__)


@register_model("calc_roberta_span")
class CalculationModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.args = args

    @classmethod
    def build_model(cls, args, task):

        """Build a new model instance."""
        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        model = cls(args, encoder)
        model.classification_heads["operation"] = OperationHead(
            args.encoder_embed_dim,
        )
        model.classification_heads["span"] = SpanHead(
            args.encoder_embed_dim,
            # other stuff too
        )

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # upgrade children modules
        super(RobertaModel, self).upgrade_state_dict_named(state_dict, name)

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


@register_model_architecture("calc_roberta_span", "calc_roberta_span")
def base_architecture(args):
    roberta_model.base_architecture(args)


@register_model_architecture("calc_roberta_span", "calc_roberta_span_base")
def roberta_span_base_architecture(args):
    roberta_model.roberta_base_architecture(args)


@register_model_architecture("calc_roberta_span", "calc_roberta_span_large")
def roberta_span_large_architecture(args):
    roberta_model.roberta_large_architecture(args)


# HARD PART - needs features, and indices of the features (just the numbers)
class SpanHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        inner_dim = 2
        self.dense = nn.Linear(input_dim, inner_dim)

    def forward(self, features):
        # forward gets the pairs, lots of stuff (decide what the shape is, etc)

        # shape of features is batch_size X sequence_length (1) X twice size of embeddings (2048)
        logits = self.dense(features)

        # return raw logits and unbind in the downstream codes
        return logits


class OperationHead(nn.Module):
    """Head for pointing start/end position of span"""

    def __init__(self, input_dim):
        super().__init__()
        # B: beginging of span, I inside span, O: outside of span         BIO / IO
        # self.num_heads = 7
        inner_dim = 5
        # self.heads = [nn.Linear(input_dim, inner_dim) for i in range(7)]
        self.dense = nn.Linear(input_dim, inner_dim)

    def forward(self, features, **kwargs):
        # shape of features is batch_size X sequence_length (1) X shape_of_embeddings (1024)
        # in this case we assume features have length 1 since just CLS
        logits = self.dense(features[:, 0, :])

        # return raw logits and unbind in the downstream codes
        return logits

        # # shape of logits is batch_size X sequence_length X 2 (one for each position)

        # unstacked_logits = torch.unbind(logits, dim=2)

        # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
        # return start_logits, end_logits
