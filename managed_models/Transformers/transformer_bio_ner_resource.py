import logging
import os
from timeit import default_timer
import torch
from flask_restful import reqparse
from flask_restful import Resource
import numpy as np

CDR_TAG_CODES = {'disease': 1, 'chemical': 2}
MUTATION_TAG_CODES = {'mutation': 1}
GENE_TAG_CODES = {'gene': 1}
CELL_LINE_TAG_CODES = {'cell_line': 1}


def batch_predict(model, tokenizer, batch_ctx, tag_codes):
    batch_tokens = tokenizer(batch_ctx, padding=True, add_special_tokens=True, return_tensors="pt")
    idx = 0
    tag_code_to_type = ['None']
    for tag_type, tag_code in tag_codes.items():
        tag_code_to_type.append(tag_type)
    with torch.no_grad():
        logits = model(**batch_tokens.to(model.device)).logits
        predicted = logits.argmax(-1).cpu().detach().numpy()
        batch_tags = []
        for idx, ctx in enumerate(batch_ctx):
            tags = {}
            for tag_type, tag_code in tag_codes.items():
                tags[tag_type] = []
            tag = []
            if np.count_nonzero(predicted[idx]) > 0:
                for t_idx, t in enumerate(predicted[idx]):
                    if t != 0:
                        if t_idx - 1 >= 0 and t_idx - 1 < len(ctx):
                            tag.append(ctx[t_idx - 1])
                    else:                        
                        if len(tag) > 0 and t_idx - 1 >= 0:
                            tag_code = predicted[idx][t_idx - 1]
                            start_idx = t_idx - 1
                            end_idx = t_idx - 1
                            while start_idx >= 0 and predicted[idx][start_idx] == tag_code:
                                start_idx = start_idx - 1
                                
                            if start_idx < t_idx - 1:
                                while start_idx >= 0 and ctx[start_idx] not in [' ', '-']:
                                    start_idx = start_idx - 1
                            
                            while end_idx < len(ctx) and ctx[end_idx] not in [' ', '.', ';', '?', ':', ',']:
                                end_idx = end_idx + 1
                            joined_tag = ''.join(ctx[start_idx + 1:end_idx])
                            joined_tag = joined_tag.strip()
                            if len(joined_tag) > 1:
                                tags[tag_code_to_type[tag_code]].append(joined_tag)
                            tag = []
            batch_tags.append(tags)
        return batch_tags

def infer_batcher(c, n):     
    # looping till length l
    for i in range(0, len(c), n):
        yield c[i:i + n]

def predict(model, tokenizer, contexts, tag_codes):
  predicted = []
  for batch_ctx in infer_batcher(contexts, 8):    
      batch_predicted = batch_predict(model, tokenizer, batch_ctx, tag_codes)
      predicted.extend(batch_predicted)
  return predicted

def invert_tags(tags, holder):
    for key, vals in tags.items():
        for val in vals:
          if val not in holder:
              holder[val] = []
          holder[val].append(key)

class BioNERResource(Resource):
    def __init__(
        self,
        cdr_tokenizer,
        cdr_model,
        mutation_tokenizer,
        mutation_model,
        gene_tokenizer,
        gene_model,
        cell_tokenizer,
        cell_model,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.post_req_parser = reqparse.RequestParser()
        self.post_req_parser.add_argument(
            "texts",
            type=str,
            action="append",
            help="list of sentences to tag",
        )
        self.cdr_tokenizer = cdr_tokenizer
        self.cdr_model = cdr_model
        self.mutation_tokenizer = mutation_tokenizer
        self.mutation_model = mutation_model
        self.gene_tokenizer = gene_tokenizer
        self.gene_model = gene_model
        self.cell_tokenizer = cell_tokenizer
        self.cell_model = cell_model
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
            contexts = request.pop("texts") or [] 
            # contexts.sort(key=lambda x: len(x))       
            if len(contexts) > 0:
                gpu_time = default_timer()
                self.logger.info(f"Inference bio_ner requested for: {len(contexts)} requests")
                cdr_outputs = predict(self.cdr_model, self.cdr_tokenizer, contexts, CDR_TAG_CODES)
                mutation_outputs = predict(self.mutation_model, self.mutation_tokenizer, contexts, MUTATION_TAG_CODES)
                gene_outputs = predict(self.gene_model, self.gene_tokenizer, contexts, GENE_TAG_CODES)
                cell_outputs = predict(self.cell_model, self.cell_tokenizer, contexts, CELL_LINE_TAG_CODES)
                for idx, ctx in enumerate(contexts):
                  output = {}
                  mut = mutation_outputs[idx]
                  cdr = cdr_outputs[idx]
                  gene = gene_outputs[idx]
                  cell = cell_outputs[idx]
                  invert_tags(mut, output)
                  invert_tags(gene, output)
                  invert_tags(cdr, output)
                  invert_tags(cell, output)
                  out_list = [[k, v] for k,v in output.items()]
                  outputs.append(out_list)
                # outputs = cdr_outputs
                gpu_time = (default_timer() - gpu_time) * 1000
                self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")
            else:
                self.logger.error(f"no inference requests received")

        except Exception as e:
            self.logger.error("Error while running predictions, err: %s" % e)
            raise e

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference bio_ner for {len(contexts)} requests finished in {wall_time:.2f}ms, {wall_time/max(1,len(contexts)):.2f}ms per sample",
        )
        res = {
            "data": outputs
        }
        return res

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
