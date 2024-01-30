import torch

from .transformer_classification_resource import TransformersClassificationResource
from .trasnformer_manager import TransformersManager
from cross_encoder.model import RetModel


def Cross_encoder_load_func(model_path, checkpoint_file, encoder):
    state_dict = torch.load(model_path + checkpoint_file)
    model = RetModel(encoder)
    print(model.load_state_dict(state_dict))
    return model.eval()


class CrossEncoderResource(TransformersClassificationResource):
    """
    BoolQ requires spacy NLP to preprocess the questions into statement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="CrossEncoder", *args, **kwargs)


class CrossEncoderManager(TransformersManager):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manager(cls, *args, **kwargs):
        return super().get_manager(
            load_func=Cross_encoder_load_func,
            head="sentence_classification_head",
            *args,
            **kwargs,
        )
