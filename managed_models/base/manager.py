"""
This Util is borrowed from https://github.com/ShannonAI/service-streamer under Apache 2.0 License

It uses spawn to start a seperate process to managing the model for each GPU. e.g. Load Balancing.
"""
import logging
import threading
import time
import sys
import weakref
from queue import Empty

import torch
import torch.multiprocessing as multiprocessing
import logging
from .master import Master
from .task import Task
from .worker import Worker


TIMEOUT = 30  # job pull timeout
TIME_SLEEP = 1 * 1e-3  # pull every 1ms
WORKER_TIMEOUT = 3600  # worker maximum running time


class _FutureCache(dict):
    "Dict for weakref only"
    pass


class Manager:
    def __init__(
        self,
        worker_num=1,  # number of workers
        cuda_devices=None,  # working cuda devices
        mp_start_method="spawn",
        active_learning=True,
        managed_model=None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self._task_id = 0
        self._worker_future_cache = _FutureCache()
        self._master_future_cache = _FutureCache()

        self.managed_model = managed_model

        self.lock = threading.Lock()

        self.worker_num = worker_num
        self.cuda_devices = cuda_devices or range(
            min(worker_num, torch.cuda.device_count()),
        )
        self.active_learning_flag = active_learning
        self.logger.info(f"Active learning: {active_learning}")

        # multi process starting method
        self.mp = multiprocessing.get_context(mp_start_method)

    def start(self, model_class, model, head=None, wait_for_ready=True):

        model_init_kwarg = model_class.get_model_init_kwargs(model, head)
        # start master
        if self.active_learning_flag:
            self.start_master(model_class, model_init_kwarg, wait_for_start=False)
        # start workers
        else:
            self.start_workers(model_class, model_init_kwarg, wait_for_start=False)

        if wait_for_ready:
            if self.active_learning_flag:
                self._master_ready_event.wait(WORKER_TIMEOUT)
            else:
                self._wait_for_worker_ready()

    def start_master(self, model_class, model_init_kwargs, wait_for_start=True):
        self._master_input_queue = self.mp.Queue()
        self._master_output_queue = self.mp.Queue()

        self.learning_tasks = []

        self._master = Master(
            model_class,
            self._master_input_queue,
            self._master_output_queue,
        )

        self._master_ps = None
        self._master_ready_event = None

        self.logger.info("Start master")

        ready_event = self.mp.Event()

        kwargs = {
            "ready_event": ready_event,
            "model_init_kwargs": model_init_kwargs,
        }

        self.logger.info("starting master")

        p = self.mp.Process(
            target=self._master.run_forever,
            kwargs=kwargs,
            name="master",
            daemon=True,
        )
        p.start()
        self._master_ps = p
        self._master_ready_event = ready_event

        if wait_for_start:
            self.logger.info("waiting for master to start")
            self._worker_ready_event.wait(WORKER_TIMEOUT)

        self.master_back_thread = threading.Thread(
            target=self._loop_collect_master_result,
            name="thread_collect_master_result",
        )
        self.master_back_thread.daemon = True

        self.master_back_thread.start()

    def start_workers(self, model_class, model_state_dict, wait_for_start=True):
        self.logger.info("Start workers")

        self._worker_input_queue = self.mp.Queue()
        self._worker_output_queue = self.mp.Queue()

        self._worker = Worker(
            model_class,
            self._worker_input_queue,
            self._worker_output_queue,
        )

        self._worker_ps = []
        self._worker_ready_events = []
        self._worker_terminate_events = []
        self._worker_destroy_events = []

        for i in range(self.worker_num):
            self._start_worker(model_state_dict, i)

        if wait_for_start:
            self.logger.info("waiting for workers to start")
            self._wait_for_worker_ready()

        self.worker_back_thread = threading.Thread(
            target=self._loop_collect_worker_result,
            name="thread_collect_worker_result",
        )
        self.worker_back_thread.daemon = True

        self.worker_back_thread.start()

    def update_workers(self, force_update=False):
        if not self.active_learning_flag:
            raise ValueError(
                "Please export ACTIVE_LEARNING=true to enable active learning features",
            )

        task_id = self._master_input("get_model_init_kwargs", {})
        future = self._master_future_cache[task_id]
        model_init_kwargs = future.result()
        self._restart_workers(model_init_kwargs)

        return future

    def restart_workers(self, model_class, model, head=None):
        if model:
            model_init_kwarg = model_class.get_model_init_kwargs(model, head)
            self._restart_workers(model_init_kwarg)

    def _restart_workers(self, model):
        self.logger.info("update workers")
        for i in range(self.worker_num):
            self.logger.info(f"updating worker {i}")

            # destroy old worker
            self.logger.info(f"sending destroy signal to worker {i}")
            destroy_event = self._worker_destroy_events.pop(0)
            destroy_event.set()

            # wait for terminate
            self.logger.info(f"waiting for worker {i} to stop")
            terminate_event = self._worker_terminate_events.pop(0)
            terminate_event.wait(WORKER_TIMEOUT)

            # remove worker process
            self._worker_ps.pop(0)

            # start a new worker
            self.logger.info(f"starting new worker with worker_id {i}")
            self._start_worker(model, i)

            # start a new worker
            self.logger.info(f"waiting new worker with worker_id {i}")
            ready_event = self._worker_ready_events[-1]
            ready_event.wait(WORKER_TIMEOUT)
            # start a new worker
            self.logger.info(f"worker worker_id {i} is ready")

    def _start_worker(
        self,
        model_init_kwargs,
        worker_id,
    ):
        ready_event = self.mp.Event()
        destroy_event = self.mp.Event()
        terminate_event = self.mp.Event()
        kwargs = {
            "worker_id": worker_id,
            "gpu_id": self.cuda_devices[worker_id % len(self.cuda_devices)],
            "ready_event": ready_event,
            "destroy_event": destroy_event,
            "terminate_event": terminate_event,
            "model_init_kwargs": model_init_kwargs,
        }

        self.logger.info(f"starting worker {worker_id}")

        p = self.mp.Process(
            target=self._worker.run_forever,
            kwargs=kwargs,
            name="worker",
            daemon=True,
        )
        p.start()
        self._worker_ps.append(p)
        self._worker_ready_events.append(ready_event)
        self._worker_destroy_events.append(destroy_event)
        self._worker_terminate_events.append(terminate_event)

    def _wait_for_worker_ready(self):
        # wait for all workers finishing init
        for (i, e) in enumerate(self._worker_ready_events):
            # todo: select all events with timeout
            is_ready = e.wait(WORKER_TIMEOUT)
            self.logger.info(f"Worker:{i} ready state: {is_ready}")

    def _loop_collect_worker_result(self):
        while True:
            message = self._recv_worker_response(timeout=TIMEOUT)
            if message:
                (task_id, item) = message
                future = self._worker_future_cache[task_id]
                future._assign_result(item)
            else:
                # todo
                time.sleep(TIME_SLEEP)

    def predict(self, **kwargs):
        """
        submit a job to workers
        """
        if self.active_learning_flag:
            task_id = self._master_input("predict", kwargs)
            future = self._master_future_cache[task_id]
            return future
        else:    
            task_id = self._worker_input(kwargs)
            future = self._worker_future_cache[task_id]
            return future

    def _worker_input(self, kwargs) -> int:
        """
        input a kwargs, distribute each item to mq, return task_id
        """
        # task id in one client
        self.lock.acquire()
        task_id = self._task_id
        self._task_id += 1
        self.lock.release()
        # request id in one task

        future = Task(task_id, weakref.ref(self._worker_future_cache))
        self._worker_future_cache[task_id] = future

        self._worker_input_queue.put((task_id, kwargs))

        return task_id

    def _recv_worker_response(self, timeout):
        try:
            message = self._worker_output_queue.get(timeout=timeout)
        except Empty:
            message = None
        return message

    def master_task(self, task, **kwargs):
        """
        submit a job to masters
        """
        task_id = self._master_input(task, kwargs)
        future = self._master_future_cache[task_id]
        return future

    def _master_input(self, action, kwargs={}) -> int:
        """
        input a kwargs, distribute each item to mq, return task_id
        """
        # task id in one client
        self.lock.acquire()
        task_id = self._task_id
        self._task_id += 1
        self.lock.release()
        # request id in one task

        future = Task(task_id, weakref.ref(self._master_future_cache))
        self._master_future_cache[task_id] = future

        self._master_input_queue.put((task_id, action, kwargs))

        return task_id

    def _recv_master_response(self, timeout):
        try:
            message = self._master_output_queue.get(timeout=timeout)
        except Empty:
            message = None
        return message

    def _loop_collect_master_result(self):
        while True:
            message = self._recv_master_response(timeout=TIMEOUT)
            if message:
                (task_id, item) = message
                future = self._master_future_cache[task_id]
                future._assign_result(item)
            else:
                # todo
                time.sleep(TIME_SLEEP)

    def get_model(self, restart_checkpoint):
        raise NotImplementedError
