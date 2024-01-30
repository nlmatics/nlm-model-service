import logging
import os
from timeit import default_timer

from flask_restful import reqparse
from flask_restful import Resource
from flask_jsonpify import jsonify
import numpy as np
import torch
from managed_models.base.smart_batcher import Batcher
from managed_models.base.smart_batcher import Sample


class DPRResource(Resource):
    def __init__(
        self,
        tokenizer,
        encoder,
        representations=False,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.post_req_parser = reqparse.RequestParser()
        self.post_req_parser.add_argument(
            "sentences",
            type=str,
            action="append",
            help="list of sentences",
        )
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.representations = representations
        self.put_req_parser = reqparse.RequestParser()


    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()
        embeddings = []
        # empty numpy array


        try:
            request = self.post_req_parser.parse_args()
            # get query
            sentences = request.pop("sentences") or []   
            if len(sentences) > 0:
                gpu_time = default_timer()
                self.logger.info(f"DPR encoding requested for: {len(sentences)} requests")
                samples = []
                tokens = self.tokenizer(sentences, return_tensors="pt", padding=True).input_ids

                model_max_token_length  = 512
                for index, token in enumerate(tokens):
                    # truncate token to max length input allowed by model
                    token = token[: model_max_token_length]
                    samples.append(Sample(index, token))

                # 4096 is max a 16G GPU can handle
                batcher = Batcher(max_token_size=min(model_max_token_length*8, 4*1024))
                batches = batcher.build_smart_batches(samples)
                for idx, batch in enumerate(batches):
                    input_ids = torch.stack(batch).to(self.encoder.device)
                    embeddings.extend(self.encoder(input_ids).pooler_output.cpu().detach().numpy())
                    input_ids.to("cpu")
                    gpu_time = (default_timer() - gpu_time) * 1000
                    self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch {idx+1}/{len(batches)} of {len(batch)}")
            else:
                self.logger.error(f"no inference requests received")

        except Exception as e:
            self.logger.error("Error while running predictions, err: %s" % e)
            raise e

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"DPR encoding for {len(sentences)} requests finished in {wall_time:.2f}ms, {wall_time/max(1,len(sentences)):.2f}ms per sample",
        )
        # use msgpack if representations provided
        if self.representations:
            return {"embeddings": np.array(embeddings).tostring()}
        else:
            return jsonify({"embeddings": embeddings})

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
