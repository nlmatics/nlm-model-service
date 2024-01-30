from threading import Event


class Task:
    """
    container of a prediction task
    """

    def __init__(self, task_id, future_cache_ref, task_size=1):
        self._id = task_id
        self._size = task_size
        self._future_cache_ref = future_cache_ref
        self._outputs = None
        self._finish_event = Event()

    def result(self, destroy=True, timeout=None):
        if self._size == 0:
            self._finish_event.set()
            return []

        finished = self._finish_event.wait(timeout)

        if not finished:
            raise TimeoutError("Task: %d Timeout" % self._id)

        # remove from future_cache
        future_cache = self._future_cache_ref()
        if future_cache is not None and destroy:
            try:
                del future_cache[self._id]
            except KeyError:
                pass

        return self._outputs

    def done(self):
        if self._finish_event.is_set():
            return True

    def _assign_result(self, result):
        self._outputs = result
        self._finish_event.set()
