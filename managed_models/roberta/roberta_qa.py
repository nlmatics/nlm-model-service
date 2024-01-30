import os

from utils.model_utils import get_active_learning_checkpoint
from .roberta_manager import RobertaManager
from .roberta_span_resource import RobertaSpanResource
from roberta.span.span_utils import load_span_model


models_base_dir = os.environ["MODELS_DIR"]
base_models_dir = "/models" if not models_base_dir else models_base_dir
roberta_models_dir = os.path.join(base_models_dir, "roberta")
gpt2_encoder_json = f"{roberta_models_dir}/encoder.json"
gpt2_vocab_bpe = f"{roberta_models_dir}/vocab.bpe"


class RobertaQAResource(RobertaSpanResource):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="QA", *args, **kwargs)


class RobertaQAManager(RobertaManager):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manager(cls, *args, **kwargs):
        return super().get_manager(
            load_func=load_span_model,
            head="span",
            *args,
            **kwargs,
        )

    def get_model(self, restart_checkpoint):
        # read model from checkpoint
        model = load_span_model(
            f"{base_models_dir}/roberta/roberta.large.qa",
            restart_checkpoint if restart_checkpoint else get_active_learning_checkpoint(
                f"{base_models_dir}/roberta/roberta.large.qa/",
                "model.pt",
            ),
            head="span",
            gpt2_encoder_json=gpt2_encoder_json,
            gpt2_vocab_bpe=gpt2_vocab_bpe,
        )
        return model

