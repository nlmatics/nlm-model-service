"""
This Util is borrowed from https://github.com/ShannonAI/service-streamer under Apache 2.0 License

It uses spawn to start a seperate process to managing the model for each GPU. e.g. Load Balancing.
"""
import logging
from logging import Filter
import os
import time
import traceback
from queue import Empty


TIMEOUT = 30  # job pull timeout
TIME_SLEEP = 1 * 1e-3  # pull every 1ms
WORKER_TIMEOUT = 3600  # worker maximum running time

class LogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"{self._rank} | {record.msg}"
        return True

class Master:
    def __init__(
        self,
        model_class,
        request_queue,
        response_queue,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert callable(model_class)

        self._pid = os.getpid()
        self.model_class = model_class

        self._request_queue = request_queue
        self._response_queue = response_queue

    def run_forever(
        self,
        model_init_kwargs,
        ready_event,
    ):
        self._pid = os.getpid()
        self.logger.info(f"Starting master with pid: {self._pid}")
        # if it is a managed model, lazy init model after forked & set CUDA_VISIBLE_DEVICES

        self.logger.info("Master initialize model on cpu")
        self.model = self.model_class()

        self.model.init_master(**model_init_kwargs)

        if ready_event:
            ready_event.set()  # tell father process that init is finished

        self.logger.info("Master is ready")

        while True:
            handled = self._run_once()
            if not handled:
                # sleep if no data handled last time
                time.sleep(TIME_SLEEP)

    def _run_once(self):
        start_time = time.time()
        try:
            job = self._recv_request(timeout=TIMEOUT)
        except TimeoutError:
            # each item timeout exceed the max latency
            return False

        if not job:
            return False

        # publish results to queue
        task_id, action, kwargs = job
        if action == "active_learning":
            try:
                outputs = self.model.active_learning(**kwargs)
            except Exception as e:
                self.logger.error(f"Error while training {str(e)}")
                self.logger.error(f"Arguments to the tasks {kwargs}")
                self.logger.error(
                    f"Traceback, err: {traceback.format_exc()}",
                )
                outputs = {
                    "status": ["error"],
                }
        elif action == "predict":
            try:
                outputs = self.model.predict(**kwargs)
            except RuntimeError as e:
                self.logger.error(f"Error while predicting from transformer {str(e)}")
                self.logger.error(
                    f"Traceback, err: {traceback.format_exc()}",
                )
                outputs = {
                    "status": ["error"],
                }
        elif action == "save":
            outputs = self.model.save(**kwargs)
        elif action == "get_model_init_kwargs":
            outputs = self.model.__class__.get_model_init_kwargs(self.model.model)
        else:
            raise RuntimeError(f"{action} not registered in master")

        self._send_response(task_id, outputs)

        self.logger.debug(f"Master run_once spend: {time.time() - start_time}")

        return True

    def _recv_request(self, timeout):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, task_id, outputs):
        self._response_queue.put((task_id, outputs))
