import logging
from timeit import default_timer

from fairseq.data.data_utils import collate_tokens
from flask_restful import reqparse
from flask_restful import Resource

from managed_models.base.manager import Manager
from managed_models.base.smart_batcher import Batcher
from managed_models.base.smart_batcher import Sample


class BaseTransformersResource(Resource):
    def __init__(
        self,
        model_name="TransformersResource",
        model_manager: Manager = None,
        smart_batch_size=512,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.model_name = model_name
        self.model_manager = model_manager
        self.smart_batch_size = smart_batch_size

        self.post_req_parser = reqparse.RequestParser()
        self.put_req_parser = reqparse.RequestParser()

    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()

        tokens, request = self.parse_post_request()

        samples = []

        for index, token in enumerate(tokens):
            token = token[: 512]
            samples.append(Sample(index, token))

        if samples:
            batcher = Batcher(max_token_size=self.smart_batch_size)
            batches = batcher.build_smart_batches(samples)

            try:

                gpu_time = default_timer()
                # submit job to manager
                tasks = []

                for batch in batches:
                    task = self.model_manager.predict(
                        input_ids=collate_tokens(batch, pad_idx=1).numpy(),
                    )
                    tasks.append(task)

                # collect logits from manager
                smart_batch_logits = []
                for task in tasks:
                    for logit in task.result():
                        smart_batch_logits.append(logit)

                gpu_time = (default_timer() - gpu_time) * 1000
                self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")

            except Exception as e:
                self.logger.error("Error while running predictions, err: %s" % e)
                raise e

            logits = batcher.restore_batch(smart_batch_logits)
        outputs = self.get_outputs_from_logits(logits, collate_tokens(tokens, pad_idx=1).numpy(), **request)

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference {self.model_name} for {len(tokens)} samples finished in {wall_time:.2f}ms, {wall_time/max(1,len(tokens)):.2f}ms per sample",
        )
        return outputs

    def put(self):
        samples, update_workers = self.parse_put_request()

        wall_time = default_timer()
        outputs = {}
        self.logger.info(
            f"Received training request for {self.model_name} with {len(samples)} samples.",
        )
        print(samples)

        for sample in samples:
            task = self.model_manager.master_task("active_learning", sample=sample)
            self.model_manager.learning_tasks.append(task)

        if update_workers:
            # wait for all learning task to finish
            for task in self.model_manager.learning_tasks:
                task.result()

            wall_time = (default_timer() - wall_time) * 1000
            self.logger.info(
                f"Training {self.model_name} for {len(samples)} samples finished in {wall_time:.2f}ms, {wall_time/max(1,len(samples)):.2f}ms per sample",
            )

            wall_time = default_timer()
            # save the model
            task = self.model_manager.master_task("save", path=self.model_manager.path)
            task.result()

            wall_time = (default_timer() - wall_time) * 1000
            self.logger.info(
                f"Saving model {self.model_name} finished in {wall_time:.2f}ms.",
            )

            wall_time = default_timer()
            # update workers
            self.model_manager.update_workers()
            outputs["update_workers"] = True

            wall_time = (default_timer() - wall_time) * 1000
            self.logger.info(
                f"Updating workers for {self.model_name} finished in {wall_time:.2f}ms.",
            )
        else:

            outputs["update_workers"] = False
            self.logger.info(
                f"Training task {self.model_name} for {len(samples)} samples is queued",
            )

        return outputs

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        raise NotImplementedError

    def parse_post_request(self, *args, **kwargs):
        raise NotImplementedError

    def parse_put_request(self, *args, **kwargs):
        raise NotImplementedError
