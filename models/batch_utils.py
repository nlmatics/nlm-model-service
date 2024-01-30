import logging
import os


class Sample:
    def __init__(self, index, token):
        self.index = index
        self.token = token


class Batcher:
    def __init__(self, max_token_size=1024):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.max_token_size = max_token_size
        try:
            self.max_token_size = int(
                self.max_token_size * float(os.getenv("MODEL_SERVER_SCALE_FACTOR", 1)),
            )
        except Exception as e:
            self.logger.error(f"Error during parsing scaled factor: {e}")
        self.logger.info(
            f"Smart batch upperbound is {self.max_token_size} tokens",
        )

    def build_smart_batches(self, samples, max_token_size=None):
        # Build samples and maintain the orders
        max_token_size = max_token_size or self.max_token_size
        # Sort by token shape to build smart batch
        sorted_index = sorted(range(len(samples)), key=lambda x: samples[x].token.shape)

        # Build smart batching
        batches = []
        cur_batch = []

        # MAGIC NUMBER
        # 4096 = 512 * 8
        # This number is safe for 16GB GPU
        # Tested with other larger numbers but get slower

        for sample_index in sorted_index:
            sample = samples[sample_index]
            cur_token_size = sample.token.shape[0]
            # adding current token will cause overflow.
            # Using len(cur_batch) * cur_token_size because collate_tokens create the batch based on the longest sample
            if (len(cur_batch) + 1) * cur_token_size > max_token_size:
                # NOTE: since max_seq_size is 512, cur_batch will always be non-empty
                batches.append(cur_batch)
                cur_batch = []

            cur_batch.append(sample.token)
        # append reminder
        if cur_batch:
            batches.append(cur_batch)

        self.logger.info(
            f"{len(samples)} are splited into {len(batches)} smart batches",
        )
        return batches, sorted_index
