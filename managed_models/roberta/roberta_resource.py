import logging
from timeit import default_timer

from fairseq.data.data_utils import collate_tokens
from flask_restful import reqparse
from flask_restful import Resource
import torch
import numpy as np
import time

from managed_models.base.manager import Manager
from managed_models.base.smart_batcher import Batcher
from managed_models.base.smart_batcher import Sample


class BaseRobertaResource(Resource):
    def __init__(
        self,
        model_name="RobertaResource",
        model=None,
        head="classification",
        model_dir=None,
        smart_batch_size=2048,
    ):
        """Constructor used by derived classes to initialize the model
        :param model_name: name of the model
        :param model: model object
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.model = model
        self.head = head
        self.model_name = model_name
        self.smart_batch_size = smart_batch_size
        self.model_dir = model_dir

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
            token = token[: self.model.model.max_positions()]
            samples.append(Sample(index, token))

        if samples:
            batcher = Batcher(max_token_size=self.smart_batch_size)
            batches = batcher.build_smart_batches(samples)

            try:

                gpu_time = default_timer()
                # submit job to manager
                smart_batch_logits = []
                for batch in batches:
                    logits = self.model.predict(
                        head=self.head,
                        tokens=collate_tokens(batch, pad_idx=1),
                        return_logits=True,
                    )
                    logits = logits.detach().cpu()
                    smart_batch_logits.extend(logits)
                gpu_time = (default_timer() - gpu_time) * 1000
                self.logger.debug(f"GPU runtime {gpu_time:.2f}ms for current batch")

            except Exception as e:
                self.logger.error("Error while running predictions, err: %s" % e)
                raise e

            logits = batcher.restore_batch(smart_batch_logits)
                    
        outputs = self.get_outputs_from_logits(logits, tokens, **request)

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Inference {self.model_name} for {len(tokens)} samples finished in {wall_time:.2f}ms, {wall_time/max(1,len(tokens)):.2f}ms per sample",
        )
        return outputs

    def construct_training_batches(self, data):
        import numpy as np
        min_batch_size = 64
        def convert_batch(cur_batch):
                batch = None
                batch_tokens = []
                if data['model'] == "span":
                    start_positions = []
                    end_positions = []
                    for batch_sample in cur_batch:
                        batch_tokens.append(batch_sample['tokens'])
                        start_positions.append(batch_sample['start_position'])
                        end_positions.append(batch_sample['end_position'])    
                    # self.logger.info("start_positions", start_positions)            
                    batch = {
                        "net_input": {
                            "src_tokens": collate_tokens(
                                batch_tokens,
                                pad_idx=1,
                            ).to(device)
                        },
                        "start_positions" : torch.as_tensor(start_positions).to(device),
                        "end_positions" : torch.as_tensor(end_positions).to(device),
                        "ntokens" : 1
                    }                    
                elif data["model"] == "classification":
                    targets = []
                    for batch_sample in cur_batch:
                        batch_tokens.append(batch_sample['tokens'])
                        targets.append(batch_sample['target'])
                    batch = {
                        "net_input": {
                            "src_tokens": collate_tokens(
                                batch_tokens,
                                pad_idx=1,
                            ).to(device)
                        },
                        "target" : torch.as_tensor(targets).to(device),
                        "ntokens" : 1
                    }
                return batch
        samples = []
        for sample_idx, tokens in enumerate(data['tokens']):
            sample = {'tokens': tokens}
            if 'end_positions' in data:
                sample['end_position'] = data['end_positions'][sample_idx]
            if 'start_positions' in data:
                sample['start_position'] = data['start_positions'][sample_idx]
            if 'target' in data:
                sample['target'] = data['target'][sample_idx]
            samples.append(sample)

        self.sorted_index = sorted(
            range(len(samples)),
            key=lambda x: samples[x]['tokens'].shape,
        )

        # np.random.shuffle(idx)
        sample_idx = []
        if torch.cuda.is_available():  
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        batches = []
        cur_batch = []
        # MAGIC NUMBER
        # 4096 = 512 * 8
        # This number is safe for 16GB GPU
        # Tested with other larger numbers but get slower
        max_tokens_in_batch = 2048#3072
        max_token_length = self.model.model.max_positions()
        for sample_index in self.sorted_index:
            sample = samples[sample_index]
            cur_token_size = sample['tokens'].shape[0]
            if cur_token_size > max_token_length:
                self.logger.error("token size is too large, truncating")
                sample['tokens'] = sample['tokens'][:max_token_length]
                cur_token_size = max_token_length
            # adding current token will cause overflow.
            # Using len(cur_batch) * cur_token_size because collate_tokens create the batch based on the longest sample
            if (len(cur_batch) + 1) * cur_token_size > max_tokens_in_batch:
                # NOTE: since max_seq_size is 512, cur_batch will always be non-empty
                batches.append(convert_batch(cur_batch))
                cur_batch = []
            cur_batch.append(sample)
        # append remainder
        if cur_batch:
            batches.append(convert_batch(cur_batch))

        return batches

    def active_learning(self, data, head, epoch=1, lr=1e-6):
        if head == "sentence_classification_head":
            lr = 1e-6
        else:
            lr = 3e-6
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        losses = []
        status = []
        # DO NOT REMOVE THE LINE BELOW - need for multithreaded logging    
        # logging.info(f"starting active learning ...")
        self.logger.info(f"starting active learning on device {self.model.device} ...")  
        # self.logger.info(f"model: {self.model}")
        # self.logger.info(f"model: {self.model.model.max_positions}")
        # self.logger.info(f"model: {self.model.model}")  
        criterion = self.model.task.build_criterion(self.model.cfg.criterion)
        self.model.train()
        criterion.train()

        for epoch_num in range(epoch):
            self.logger.info(f"\n*** training epoch {epoch_num} ***")
            total_correct = 0
            total_sentences = 0
            status = []
            batches = self.construct_training_batches(data)
            self.logger.info(f"total batches: {len(batches)} from {len(data['tokens'])} samples")
            for batch_idx, batch in enumerate(batches):
                self.logger.info(f"processing batch: {batch_idx} with token size {batch['net_input']['src_tokens'].shape}")
                total_loss, sample_size, logging_output = criterion.forward(
                    self.model.model,
                    batch,
                )

                total_loss.backward()

                losses.append(np.array(total_loss.data.cpu().detach().numpy()))
                optimizer.step()
                # batch['net_input']['src_tokens'].to("cpu")
                total_correct += logging_output['ncorrect'].item()
                total_sentences += logging_output['nsentences']
                self.logger.info(logging_output)
                if 'corrects' in logging_output:
                    status.extend(np.array(logging_output['corrects'].cpu().detach().numpy(), dtype=bool)) 
                self.logger.info(f"batch {batch_idx} --> correct: {logging_output['ncorrect'].item()} loss: {total_loss}")

            percent_correct = total_correct / total_sentences    

            self.logger.info(f"Epoch {epoch_num} --> correct: {total_correct}, total: {total_sentences}, %correct: {percent_correct}, loss: {total_loss}")
            if percent_correct > 0.8:
                self.logger.info(f"Exiting training because 80% accuracy is achieved")
                break 
            
        # self.logger.info(f"Model optimzed: status: {status} losses: {losses}")
        self.model.eval()
        criterion.eval()
        return losses, status

    def save(self, path):
        if path is None:
            self.logger.error("need to set path to save the model")
            return
        path = f"{path}/active_learning-{time.time()}.pt"
        self.model.save(path)
        self.model.cuda().eval()
        return path

    def put(self):
        wall_time = default_timer()
        outputs = {}
        samples, save_model, restart_workers = self.parse_put_request()

        if len(samples) > 0 and 'tokens' in samples[0] and len(samples[0]['tokens']) > 0 :
            self.logger.info(
                f"Received training request for {self.model_name} with with {len(samples[0]['tokens'])} samples.",
            )
            # there is only one sample, sample[0]
            # sample[0]['tokens'] is a list of tokens
            # sample[0]['start_positions'] is a list of start positions
            # sample[0]['end_positions'] is a list of end positions
            # sample[0]['target'] is a list of targets 
            # print("samples: ", samples)
            for sample in samples:
                loss, status = self.active_learning(sample, self.head, epoch=1, lr=1e-6)
                self.logger.info(
                    f"result of training is status: {status}, loss: {loss}"
                )

                outputs['status'] = [bool(item) for item in status]
                wall_time = (default_timer() - wall_time) * 1000
                self.logger.info(
                    f"Training {self.model_name} for {len(samples)} samples finished in {wall_time:.2f}ms, {wall_time/max(1,len(samples)):.2f}ms per sample",
                )
        else:
            self.logger.info(
                f"Ignoring empty request for training.",
            )
        # self.logger.info(f"returning outputs: ", outputs['status'])
        if save_model:
            self.logger.info(
                f"Saving model..in dir {self.model_dir}",
            )

            wall_time = default_timer()
            # save the model
            save_path = self.save(path=self.model_dir)
            outputs['save at path'] = save_path

            wall_time = (default_timer() - wall_time) * 1000

            self.logger.info(
                f"Saving model {self.model_name} in {save_path} finished in {wall_time:.2f}ms.",
            )

        return outputs

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        raise NotImplementedError

    def parse_post_request(self, *args, **kwargs):
        raise NotImplementedError

    def parse_put_request(self, *args, **kwargs):
        raise NotImplementedError
