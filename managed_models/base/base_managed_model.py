import logging
import os

import torch

from .master import Master


class BaseManagedModel:
    def __init__(self, gpu_id=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.gpu_id = gpu_id
        self.set_gpu_id(self.gpu_id)

    def set_gpu_id(self, gpu_id=None):
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

    def init_worker(self):
        """
        function to load the worker models to self.model
        """
        raise NotImplementedError

    def init_master(self):
        """
        function to load the master models to self.model
        """
        raise NotImplementedError

    def predict(self):
        """
        function to predict the job
        """
        raise NotImplementedError

    def active_learning(self):
        """
        function to learn the job and labels
        """
        raise NotImplementedError

    def save(self):
        """
        function to save the model to path
        """
        raise NotImplementedError

    def get_model_init_kwargs(self):
        """
        function to generate the worker init kwargs that will be used in init_workers
        """
        raise NotImplementedError

    def serve(self, worker_num=None):
        if not worker_num:
            worker_num = torch.cuda.device_count()

        return Master(
            self,
            worker_num=worker_num,
        )
