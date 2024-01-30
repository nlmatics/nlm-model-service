from transformers import RobertaModel
from transformers import AutoTokenizer

from .transformer_managed_model import CrossEncoderManagedModel
from managed_models.base.manager import Manager


class TransformersManager(Manager):
    """
    Modify Manager before start the master and workers
    """

    def start(self, model_class, model, head, path, *args, **kwargs):
        """
        clone attribute from model to manager
        """
        self.encode = kwargs["tok_encoder"]
        self.decode = kwargs["tok_decoder"]
        self.max_positions = 512
        self.head = head
        self.path = path

        # create map between label and predicted id
        self.label2id = {}
        self.id2label = []
        """
        for label in model.task.label_dictionary.symbols:
            if label.startswith(("<", "madeupword")) or label.isdigit():
                continue
            self.label2id[label] = len(self.label2id)
            self.id2label.append(label)

        self.logger.info(
            f"{model_class.__name__} has {len(self.id2label)} labels: {self.id2label}",
        )
        """
        # start deamons
        super().start(model_class, model, head = self.head)

    @classmethod
    def get_manager(
        cls,
        model_name_or_path,
        checkpoint_file,
        head,
        encoder=RobertaModel.from_pretrained("roberta-large"),
        worker_num=1,
        active_learning=False,
        load_func=None,
    ):
        # build managed model
        manager = cls(
            worker_num=worker_num,
            active_learning=active_learning,
            managed_model=CrossEncoderManagedModel,
        )

        manager.logger.info(
            f"loading model from {model_name_or_path}/{checkpoint_file}",
        )

        # read model from checkpoint
        model = load_func(
            model_name_or_path,
            checkpoint_file,
            encoder,
        )

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        manager.start(
            model_class=manager.managed_model,
            tok_encoder=tokenizer,
            tok_decoder=tokenizer.decode,
            model=model,
            head=head,
            path=model_name_or_path,
        )

        # clear model weight to release memory
        model.model = None

        return manager
