"""
This Util is borrowed from https://github.com/ShannonAI/service-streamer under Apache 2.0 License

It uses spawn to start a seperate process to managing the model for each GPU. e.g. Load Balancing.
"""
import logging
import os
import time
import traceback
from queue import Empty


TIMEOUT = 30  # job pull timeout
TIME_SLEEP = 1 * 1e-3  # pull every 1ms


class Worker:
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
        worker_id=None,
        gpu_id=None,
        model_init_kwargs=None,
        ready_event=None,
        destroy_event=None,
        terminate_event=None,
    ):
        self._worker_id = worker_id
        self._pid = os.getpid()
        self.logger.info(f"Starting worker {self._worker_id} with pid: {self._pid}")
        # if it is a managed model, lazy init model after forked & set CUDA_VISIBLE_DEVICES

        self.logger.info(f"Worker {self._worker_id} initialize model on gpu:{gpu_id}")
        self.model = self.model_class(gpu_id)
        self.model.set_gpu_id(gpu_id)

        self.model.init_worker(**model_init_kwargs)

        if ready_event:
            ready_event.set()  # tell father process that init is finished

        if destroy_event:
            self._destroy_event = destroy_event

        if terminate_event:
            self._terminate_event = terminate_event

        self.logger.info(f"Worker {self._worker_id} is ready")

        while True:
            handled = self._run_once()
            if self._destroy_event and self._destroy_event.is_set():
                self.logger.info(f"Worker {self._worker_id} received exit signal")
                break
            if not handled:
                # sleep if no data handled last time
                time.sleep(TIME_SLEEP)

        # delete model, release gpu memory
        del self.model
        # tell master this worker is stoped
        self.logger.info(f"Worker {self._worker_id} shutdown")
        self._terminate_event.set()

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
        task_id, model_input = job

        try:
            model_outputs = self.model.predict(**model_input)
        except RuntimeError as e:
            self.logger.error(f"Error while predicting from transformer {str(e)}")
            self.logger.error(
                f"Traceback, err: {traceback.format_exc()}",
            )
            model_outputs = []

        self._send_response(task_id, model_outputs)

        self.logger.debug(
            f"[gpu worker {self._worker_id}] run_once spend: {time.time() - start_time}",
        )
        return True

    def _recv_request(self, timeout):
        try:
            item = self._request_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError
        else:
            return item

    def _send_response(self, task_id, model_output):
        self._response_queue.put((task_id, model_output))
