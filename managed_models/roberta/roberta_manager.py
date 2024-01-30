from fairseq import tasks

from managed_models.base.manager import Manager
from managed_models.roberta.roberta_managed_model import RobertaManagedModel


class RobertaManager(Manager):
    """
    Modify Manager before start the master and workers
    """

    def start(self, model_class, model, head, path, *args, **kwargs):
        """
        clone attribute from model to manager
        """
        self.cfg = model.cfg
        self.task = tasks.setup_task(self.cfg.task)
        self.encode = model.encode
        self.decode = model.decode
        self.max_positions = model.model.max_positions
        self.head = head
        self.path = path

        # create map between label and predicted id
        self.label2id = {}
        self.id2label = []

        for label in model.task.label_dictionary.symbols:
            if label.startswith(("<", "madeupword")) or label.isdigit():
                continue
            self.label2id[label] = len(self.label2id)
            self.id2label.append(label)

        self.logger.info(
            f"{model_class.__name__} has {len(self.id2label)} labels: {self.id2label}",
        )

        # start deamons
        super().start(model_class, model)

    @classmethod
    def get_manager(
        cls,
        model_name_or_path,
        checkpoint_file,
        head,
        gpt2_encoder_json,
        gpt2_vocab_bpe,
        worker_num=1,
        active_learning=False,
        load_func=None,
    ):
        # build managed model
        manager = cls(
            worker_num=worker_num,
            active_learning=active_learning,
            managed_model=RobertaManagedModel,
        )

        manager.logger.info(
            f"loading model from {model_name_or_path}/{checkpoint_file}",
        )

        # read model from checkpoint
        model = load_func(
            model_name_or_path,
            checkpoint_file,
            head=head,
            gpt2_encoder_json=gpt2_encoder_json,
            gpt2_vocab_bpe=gpt2_vocab_bpe,
        )
        model.eval()
        manager.start(
            model_class=manager.managed_model,
            model=model,
            head=head,
            path=model_name_or_path,
        )

        # clear model weight to release memory
        model.model = None

        return manager

    def get_model(self, restart_checkpoint):
        return None

    def restart_workers(self, model_class, model, head=None):
        super().restart_workers(model_class, model, head)

