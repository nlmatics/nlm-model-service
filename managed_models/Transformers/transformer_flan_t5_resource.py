import logging
import os
from timeit import default_timer
import torch

from flask_restful import reqparse, Resource, inputs
from managed_models.base.smart_batcher import Batcher
from managed_models.base.smart_batcher import Sample

TOKEN_SIZE_PER_BATCH = 512
class FlanT5Resource(Resource):
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
            "prompts",
            type=str,
            action="append",
            help="list of prompts",
        )
        self.post_req_parser.add_argument(
            "max_length",
            type=int,
            help="max length of output",
        )
        self.tokenizer = tokenizer
        self.model = model
        self.put_req_parser = reqparse.RequestParser()


    def post(self):
        """Handles post requests to run inference
        :return: the predictions for on the provided input
        """
        wall_time = default_timer()
        confidences = []
        output_sequences = []
        try:
            request = self.post_req_parser.parse_args()
            # get query
            prompts = request.pop("prompts") or []      
            max_length = inputs.positive(request.pop("max_length")) or 200

            
            if len(prompts) > 0:

                gpu_time = default_timer()
                self.logger.info(f"Inference requested for: {len(prompts)} prompts")
                samples = []
                tokens = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids
                model_max_token_length  = self.model.shared.embedding_dim
                for index, token in enumerate(tokens):
                    # truncate token to max length input allowed by model
                    token = token[: model_max_token_length]
                    samples.append(Sample(index, token))

                # 4096 is max a 16G GPU can handle
                batcher = Batcher(max_token_size=min(model_max_token_length*8, 4*1024))
                batches = batcher.build_smart_batches(samples)
                for idx, batch in enumerate(batches):
                    input_ids = torch.stack(batch).to(self.model.device)
                    model_output = self.model.generate(input_ids, max_length=max_length, output_scores=True, return_dict_in_generate=True)

                    confidences.extend(torch.softmax(model_output.scores[0], dim=-1).max(dim=1).values.tolist())
                    output_sequences.extend(self.tokenizer.batch_decode(model_output.sequences, skip_special_tokens=True))
                    input_ids.to("cpu")
                    gpu_time = (default_timer() - gpu_time) * 1000
                    self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch {idx+1}/{len(batches)} of {len(batch)}")

                gpu_time = (default_timer() - gpu_time) * 1000
                self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for {len(samples)} items")
            else:
                self.logger.error(f"no inference requests received")

        except Exception as e:
            self.logger.error("Error while running predictions, err: %s" % e)
            raise e

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference t5 for {len(prompts)} prompts finished in {wall_time:.2f}ms, {wall_time/max(1,len(prompts)):.2f}ms per sample",
        )
        return {"outputs": output_sequences, "confidences": confidences}

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
