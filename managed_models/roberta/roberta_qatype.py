import os

from fairseq.models.roberta.model import RobertaModel

from utils.model_utils import get_active_learning_checkpoint
from .roberta_classification_resource import RobertaClassificationResource
from .roberta_manager import RobertaManager

models_base_dir = os.environ["MODELS_DIR"]
base_models_dir = "/models" if not models_base_dir else models_base_dir
roberta_models_dir = os.path.join(base_models_dir, "roberta")
gpt2_encoder_json = f"{roberta_models_dir}/encoder.json"
gpt2_vocab_bpe = f"{roberta_models_dir}/vocab.bpe"


class RobertaQATypeResource(RobertaClassificationResource):
    def __init__(self, *args, **kwargs):
        super().__init__(model_name="QAType", *args, **kwargs)

    def preprocess_sentence(self, text):
        text = text.lower().strip()
        if text.endswith("?"):
            text = text[:-1]
        return text


class RobertaQATypeManager(RobertaManager):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manager(cls, *args, **kwargs):
        return super().get_manager(
            load_func=RobertaModel.from_pretrained,
            head="sentence_classification_head",
            *args,
            **kwargs,
        )

    def get_model(self, restart_checkpoint):
        # read model from checkpoint
        model = RobertaModel.from_pretrained(
            f"{base_models_dir}/roberta/roberta.large.qatype.lower.RothWithQ",
            restart_checkpoint if restart_checkpoint else get_active_learning_checkpoint(
                f"{base_models_dir}/roberta/roberta.large.qatype.lower.RothWithQ/",
                "model.pt",
            ),
            head="sentence_classification_head",
            gpt2_encoder_json=gpt2_encoder_json,
            gpt2_vocab_bpe=gpt2_vocab_bpe,
        )
        return model
