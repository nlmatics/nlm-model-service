import logging
from models.parallel_utils import predict_smart_batch
from timeit import default_timer
from tqdm import tqdm

import numpy as np
import torch
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource

from fairseq.data.data_utils import collate_tokens
from models.batch_utils import Batcher
from models.batch_utils import Sample
from sklearn.metrics.pairwise import cosine_similarity


class BaseEncoder(Resource):
    def __init__(self, name, model, representations=None, **kwargs):
        """Serve huggingface.transformers models for classification tasks

        This model should be able to handle all classification tasks using transformer models

        Args:
            name: a str of model's name
            model: a instance of the model to serve
            tokenizer: the corresponding tokenizer of the model
            batch_size: batch_size of the model

        Returns:
            A instance of the flask resource
        """
        # init flask resource to wrap new response data if provided
        self.representations = representations
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.name = name
        self.model = model

        # argument parser
        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "sentences",
            type=str,
            action="append",
            help="First list of sentences",
        )
        self.req_parser.add_argument(
            "headers",
            type=str,
            action="append",
            required=False,
        )
        self.req_parser.add_argument(
            "batch_size",
            type=int,
            action="append",
            help="Second list of sentences",
        )

    def post(self):
        """Handle post requests to run inference

        Returns:
            json: the jsonified predictions for on the provided input
        """

        args = self.req_parser.parse_args()
        sentences = args["sentences"]
        headers = args["headers"]

        self.logger.info(f"Start encoding {self.name} {len(sentences)} samples")
        wall_time = default_timer()

        try:
            emb = self.encode(sentences, headers=headers)
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
        # throw exception
        except Exception as e:
            self.logger.error(f"Failed to encode {self.name}: {e}", exc_info=True)
            msg = {"status": "fail", "message": str(e), "embeddings": None}
            if self.representations:
                return msg
            else:
                return jsonify(msg)

        wall_time = (default_timer() - wall_time) * 1000
        self.logger.info(
            f"Encoding {self.name} {len(sentences)} samples finished in {wall_time:.2f}ms, {wall_time/len(sentences):.2f}ms per sample",
        )
        try:
            # use msgpack if representations provided
            if self.representations:
                return {"embeddings": emb.tostring()}
            else:
                return jsonify({"embeddings": emb.tolist()})
        # throw exception
        except Exception as e:
            msg = {"status": "fail", "message": str(e), "embeddings": None}
            if self.representations:
                return msg
            else:
                return jsonify(msg)

    def encode(self, text) -> np.array:
        raise NotImplementedError


class DistilBERTEncoder(BaseEncoder):
    def __init__(
        self, name, model, representations=None, batch_size=8, use_gpu=True, **kwargs
    ):
        super().__init__(name, model, representations)

        self.batch_size = batch_size
        self.device = "cpu"
        if use_gpu and torch.cuda.device_count():
            self.device = "cuda"
        self.model.to(self.device)

    def encode(self, text, batch_size=None, headers=None):

        self.logger.info(f"Start encoding distilbert {len(text)} samples")

        batch_size = batch_size if batch_size else self.batch_size

        emb = np.array(
            [x for x in self.model.encode(text, batch_size=batch_size)],
        ).astype(np.float32)

        return emb


class DPREncoder(BaseEncoder):
    def __init__(self, name, model, representations=None, **kwargs):
        super().__init__(name, model, representations)
        self.tokenizer = model
        self.model = model
        self.max_length = 512

    def text_to_tensor(
        self, text: str, title: str = None, add_special_tokens: bool = True
    ):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(title, text)
        else:
            token_ids = self.tokenizer.encode(text)

        return torch.tensor(token_ids[:512])

    def get_attn_mask(self, tokens_tensor):
        return tokens_tensor != self.tokenizer.pad_token_id

    def encode(self, texts, headers=None):

        samples = []
        for index in range(len(texts)):
            token = self.text_to_tensor(
                texts[index], title=headers[index] if headers else None
            )
            samples.append(Sample(index, token))

        batches, sorted_index = Batcher().build_smart_batches(
            samples, max_token_size=2048
        )

        unsorted_embs = predict_smart_batch(
            self.model,
            batches,
        )
        sample_index = 0
        for emb in unsorted_embs:
            samples[sorted_index[sample_index]].emb = emb
            sample_index += 1

        return [x.emb for x in samples]


class SIFEncoder(BaseEncoder):
    def __init__(self, name, model, representations=None, **kwargs):
        super().__init__(name, model, representations)

    def encode(self, text, **kwargs):

        self.logger.info(f"Start encoding sif {len(text)} samples")
        emb = self.model.encode(text).astype(np.float32)

        return emb


class SIFSimilarity(Resource):
    def __init__(self, name, model, representations=False):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.name = name
        self.model = model
        self.representations = representations

        # argument parser
        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "sentences_a",
            type=str,
            action="append",
            help="First list of sentences",
        )
        self.req_parser.add_argument(
            "sentences_b",
            type=str,
            action="append",
            help="Second list of sentences",
        )
        self.req_parser.add_argument(
            "batch_size",
            type=int,
            action="append",
        )

    def cosine_similarity(self, sentences_a, sentences_b):
        emb_a = self.model.encode(sentences_a).astype(np.float32)
        emb_b = self.model.encode(sentences_b).astype(np.float32)

        sims = cosine_similarity(
            emb_a.reshape(-1, 300),
            emb_b.reshape(-1, 300),
        )
        return sims

    def post(self):
        args = self.req_parser.parse_args()
        sentences_a = args["sentences_a"]
        sentences_b = args["sentences_b"]

        self.logger.info(f"Start comparing {self.name} {len(sentences_a)} sentences with {len(sentences_b)} sentences")
        wall_time = default_timer()

        try:
            sims = self.cosine_similarity(sentences_a = sentences_a, sentences_b=sentences_b)
            wall_time = (default_timer() - wall_time) * 1000
            self.logger.info(
                f"Comparing {self.name} {len(sentences_a)} sentences with {len(sentences_b)} sentences finished in {wall_time:.2f}ms, {wall_time/(len(sentences_a) + len(sentences_b)):.2f}ms per sample",
            )
        # throw exception
        except Exception as e:
            self.logger.error(f"Failed to encode {self.name}: {e}", exc_info=True)
            msg = {"status": "fail", "message": str(e), "sims": None}
            if self.representations:
                return msg
            else:
                return jsonify(msg)

        try:
            # use msgpack if representations provided
            if self.representations:
                return {"sims": sims.tostring()}
            else:
                return jsonify({"sims": sims.tolist()})
        # throw exception
        except Exception as e:
            msg = {"status": "fail", "message": str(e), "sims": None}
            if self.representations:
                return msg
            else:
                return jsonify(msg)


class RobertaEncoder(BaseEncoder):
    def __init__(self, model, a=1e-7, layer_idx=10, cls_token_only=False, **kwargs):
        super().__init__("roberta-encoder", model)
        source_dictionary = self.model.task.source_dictionary
        self.N = 0
        self.word_weights = []
        self.layer_idx = layer_idx
        self.word_freqs = source_dictionary.count
        for word_freq in self.word_freqs:
            self.N += float(word_freq)

        for word_freq in self.word_freqs:
            word_weight = a / (a + (word_freq / self.N) ** 2)
            self.word_weights.append(word_weight)

        self.cls_token_only = cls_token_only

    def get_token_weights(self, tokens):
        weights = []
        counts = []
        for tok_idx in tokens:
            counts.append(self.word_freqs[tok_idx])
            weights.append(self.word_weights[tok_idx])
        return np.array(weights), counts

    def encode(self, sentences):

        self.logger.info(f"Start encoding roberta {len(sentences)} samples")

        samples = []
        for index in range(len(sentences)):
            tokens = self.model.encode(sentences[index])
            tokens = tokens[:512]
            samples.append(Sample(index, tokens))

        batches, sorted_index = Batcher().build_smart_batches(samples)
        encodings = []
        for batch in batches:
            weights = []
            batch_tokens = collate_tokens(
                batch,
                pad_idx=1,
            )
            with torch.no_grad():
                features = self.model.extract_features(
                    batch_tokens,
                    return_all_hiddens=True,
                )
                for batch_idx, item_toks in enumerate(batch):
                    if self.cls_token_only:
                        encoding = (
                            features[self.layer_idx][batch_idx][0]
                            .cpu()
                            .detach()
                            .numpy()
                            .tolist()
                        )
                        encodings.append(encoding)
                    else:
                        # exclude start and end markers
                        item_weights, _ = self.get_token_weights(item_toks[1:-1])
                        weights.append(item_weights)
                        # exclude padding words
                        item_features = (
                            features[self.layer_idx][batch_idx][1 : len(item_toks) - 1]
                            .cpu()
                            .detach()
                            .numpy()
                        )

                        encoding = np.average(
                            item_features,
                            axis=0,
                            weights=item_weights,
                        )
                        # normalize
                        encoding = encoding / np.linalg.norm(encoding)
                        # convert to list
                        encoding = encoding.tolist()
                        encodings.append(encoding)

        for idx, encoding in enumerate(encodings):
            samples[sorted_index[idx]].encoding = encoding

        emb = [s.encoding for s in samples]

        return emb

    def post(self):
        """Handle post requests to run inference

        Returns:
            json: the jsonified predictions for on the provided input
        """

        args = self.req_parser.parse_args()
        sentences = args["sentences"]

        try:
            emb = self.encode(sentences)
            return jsonify({"embeddings": emb})
        # throw exception
        except Exception as e:
            self.logger.error("failed to encode", exc_info=True)
            msg = {"status": "fail", "message": str(e), "embeddings": None}
            if self.representations:
                return msg
            else:
                return jsonify(msg)


# for local testing
if __name__ == "__main__":

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    server = DistilBERTEncoder("DistilBERT", model, batch_size=8)

    text = [
        "What percentage of farmland grows wheat?",
        "More than 50% of this area is sown for wheat, 33% for barley and 7% for oats.",
        "Where did the Exposition take place?",
        "This World's Fair devoted a building to electrical exhibits.",
        "",
    ]

    emb = server.encode(text)
