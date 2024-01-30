import logging
import os
from timeit import default_timer

from flask_restful import reqparse
from flask_restful import Resource

class T5Resource(Resource):
    def __init__(
        self,
        tokenizer,
        model,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.post_req_parser = reqparse.RequestParser()
        self.post_req_parser.add_argument(
            "inference_requests",
            type=str,
            action="append",
            help="list of requests",
        )
        self.tokenizer = tokenizer
        self.model = model
        self.put_req_parser = reqparse.RequestParser()


    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()
        outputs = []

        try:
            request = self.post_req_parser.parse_args()
            # get query
            inference_requests = request.pop("inference_requests") or []        
            if len(inference_requests) > 0:
                gpu_time = default_timer()
                self.logger.info(f"Inference requested for: {len(inference_requests)} requests")
                inputs = self.tokenizer(inference_requests, return_tensors="pt", padding=True)
                output_sequences = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.model.device),
                    attention_mask=inputs["attention_mask"].to(self.model.device),
                    max_new_tokens=120
                )            
                outputs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                gpu_time = (default_timer() - gpu_time) * 1000
                self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")
            else:
                self.logger.error(f"no inference requests received")

        except Exception as e:
            self.logger.error("Error while running predictions, err: %s" % e)
            raise e

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference t5 for {len(inference_requests)} requests finished in {wall_time:.2f}ms, {wall_time/max(1,len(inference_requests)):.2f}ms per sample",
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


    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        raise NotImplementedError

    def parse_post_request(self, *args, **kwargs):
        raise NotImplementedError

    def parse_put_request(self, *args, **kwargs):
        raise NotImplementedError
