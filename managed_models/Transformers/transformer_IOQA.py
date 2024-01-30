import torch

from .transformer_IO_resource import SpanResource
from .trasnformer_manager import TransformersManager
from IO_QA import QAModel


def IOQA_load_func(model_path, checkpoint_file, encoder):
    state_dict = torch.load(model_path + checkpoint_file)
    model = QAModel(encoder)
    print(model.load_state_dict(state_dict))
    return model.eval()


class IOQAResource(SpanResource):
    """
    BoolQ requires spacy NLP to preprocess the questions into statement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="IO", *args, **kwargs)


class IOQAManager(TransformersManager):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manager(cls, *args, **kwargs):
        return super().get_manager(
            load_func=IOQA_load_func,
            head="IO",
            *args,
            **kwargs,
        )
