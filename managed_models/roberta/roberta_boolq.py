import os

from fairseq.models.roberta.model import RobertaModel

from utils.model_utils import get_active_learning_checkpoint
from .roberta_classification_resource import RobertaClassificationResource
from .roberta_manager import RobertaManager
from models.boolq_utils import Question2Sentence

models_base_dir = os.environ["MODELS_DIR"]
base_models_dir = "/models" if not models_base_dir else models_base_dir
roberta_models_dir = os.path.join(base_models_dir, "roberta")
gpt2_encoder_json = f"{roberta_models_dir}/encoder.json"
gpt2_vocab_bpe = f"{roberta_models_dir}/vocab.bpe"


class RobertaBOOLQResource(RobertaClassificationResource):
    """
    BoolQ requires spacy NLP to preprocess the questions into statement
    """

    def __init__(self, spacy_nlp=None, *args, **kwargs):
        super().__init__(model_name="BOOLQ", *args, **kwargs)
        self.spacy_nlp = spacy_nlp
        # utils to convert questions to sentences
        # using remote call when spacy_nlp is None
        self.questions_to_statements = Question2Sentence(spacy_nlp)

    def encode(self, questions, sentences=[], keep_statement=False):
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(sentences, str):
            sentences = [sentences]

        # convert questions to statement
        if not keep_statement:
            if questions.count(questions[0]) == len(questions):
                statements = self.questions_to_statements([questions[0]]) * len(questions)
            else:
                statements = self.questions_to_statements(questions)
            # boolq input is (sentence, statement)
        else:
            if questions.count(questions[0]) == len(questions):
                statements = [questions[0]] * len(questions)
            else:
                statements = questions
            
        return super().encode(sentences, statements)


class RobertaBOOLQManager(RobertaManager):
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
            f"{base_models_dir}/roberta/roberta.large.boolq",
            restart_checkpoint if restart_checkpoint else get_active_learning_checkpoint(
                f"{base_models_dir}/roberta/roberta.large.boolq/",
                "model.pt",
            ),
            head="sentence_classification_head",
            gpt2_encoder_json=gpt2_encoder_json,
            gpt2_vocab_bpe=gpt2_vocab_bpe,
        )
        return model

