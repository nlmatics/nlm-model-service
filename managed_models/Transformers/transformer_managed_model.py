import pickle
import time

import numpy as np
import torch
from transformers import RobertaConfig
from transformers import RobertaModel

from cross_encoder.model import RetModel
from IO_QA import QAModel
from IO_relation import RelModel
from managed_models.base.base_managed_model import BaseManagedModel

# this import is to make sure the span task is registered in fairseq


class CrossEncoderManagedModel(BaseManagedModel):
    def init_worker(self, model_state_dict, head):
        self.logger.info("initializing worker")

        self.logger.debug("build_model")
        roberta = RobertaModel.from_pretrained("roberta-large")
        model = self.build_model(roberta, head)

        self.logger.debug("deserilize state_dict")
        model_state_dict = pickle.loads(model_state_dict)

        self.logger.debug("load state_dict")
        model.load_state_dict(model_state_dict)

        self.logger.info("preparing for GPU inference")
        # use fp16
        model = model.half()
        # push model to cuda
        model = model.cuda(self.gpu_id)
        # eval mode
        model = model.eval()

        self.model = model

        self.logger.info(
            f"Model {self.__class__.__name__} initialized on GPU {self.gpu_id}",
        )

    def init_master(self, model_state_dict, head):
        self.logger.info("initializing master")

        self.logger.debug("build_model")
        config = RobertaConfig.from_pretrained("roberta-large")
        roberta = RobertaModel(config)
        model = self.build_model(roberta, head)

        self.logger.debug("deserilize state_dict")
        model_state_dict = pickle.loads(model_state_dict)

        self.logger.debug("load state_dict")
        model.load_state_dict(model_state_dict)

        # init criterion
        # self.criterion = model.task.build_criterion(model.cfg.criterion)

        self.model = model

        # put master model in train
        self.model.train()

        # put criterion always in train
        # self.criterion.train()

    def criterion(self, model, sample):
        """

        sample:{
            "positive_example":
            "neg_exapmle_1":
            "neg_example_2":
        }

        """
        loss_func = torch.nn.CrossEntropyLoss()
        logits = model(sample["input"]).reshape(-1, 3)
        target = torch.ones(len(logits)).long()
        loss = loss_func(logits, target)
        print(logits, loss.item())
        return loss

    def predict(self, **kwargs):
        """
        predict the given task
        """
        """
        assert (
            "tokens" in kwargs and "head" in kwargs
        ), f"Roberta requires tokens and head for prediction, got {kwargs}"
        """
        if isinstance(kwargs["input_ids"], np.ndarray):
            kwargs["input_ids"] = torch.from_numpy(kwargs["input_ids"])

        kwargs["input_ids"] = kwargs["input_ids"].to(f"cuda:{self.gpu_id}")

        # adding attetion mask to inputs of the model ####
        mask = torch.ones(kwargs["input_ids"].shape).to(f"cuda:{self.gpu_id}")
        mask[torch.where(kwargs["input_ids"] == 1)] = 0
        kwargs["attention_mask"] = mask

        assert isinstance(kwargs["input_ids"], torch.Tensor)

        with torch.no_grad():
            logits = self.model(kwargs)

        return logits.detach().cpu()

    def active_learning(self, sample, epoch=10, lr=5e-9):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        losses = []
        print(sample)
        for _ in range(epoch):
            total_loss = self.criterion(
                self.model,
                sample,
            )

            total_loss.backward()

            losses.append(total_loss.data)
            optimizer.step()
        self.logger.info(f"Model optimzed: {total_loss}")
        return losses

    def save(self, path):
        if path is None:
            self.logger.error("need to set path to save the model")
            return
        path = f"{path}/active_learning-{time.time()}.pt"
        torch.save(self.model.state_dict(), path)
        # self.model.save(path)

    @staticmethod
    def get_model_init_kwargs(model, head):
        kwargs = {
            # "model_class": model.__class__,
            # "model_cfg": model.cfg,
            "model_state_dict": pickle.dumps(model.state_dict()),
            "head":head
        }

        return kwargs

    def build_model(self, encoder, head=None):
        if head == "IO":
            return QAModel(encoder)
        elif head == "crossencoder":
            return RetModel(encoder)
        elif head == "IORelation":
            return RelModel(encoder)
