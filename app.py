import logging
import os

import msgpack
import torch
import transformers
from fairseq.models.roberta import RobertaModel
from flask import Flask
from flask import make_response
from flask_restful import Api
from sentence_transformers import SentenceTransformer

from models import bert
from models import encoder
from models import healthcheck
from models import qnli
from models import roberta
from models import squad2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    from sif.cython_sif import SIFModel
except ImportError:
    logger.error("cython sif is not found, fall back to python implementation")
    from sif.sif import SIFModel

models_from_env = os.environ["MODELS"].split()
models_dir_from_env = os.environ["MODELS_DIR"]
gpu_enabled = os.environ["GPU_ENABLED"] if "GPU_ENABLED" in os.environ else "no"

gpu_enabled = gpu_enabled is not None and (
    gpu_enabled == 1 or gpu_enabled == "true" or gpu_enabled == "yes"
)

models_base_dir = (
    "/models"
    if not models_dir_from_env or models_dir_from_env == ""
    else models_dir_from_env
)
if not os.path.exists(models_base_dir):
    logger.error(f"models dir {models_base_dir} does not exists")
    exit(1)

registered_models = [
    "qnli",
    "mnli",
    "sts-b",
    "boolq",
    "squad",
    "albert-xlarge-v1-qnli",
    "albert-xxlarge-v1-qnli",
    "albert-xxlarge-v1-squad2",
    "albert-xlarge-v2-squad2",
    "roberta-base-squad2",
    "distilbert-encoder",
    "sif-encoder",
]
models_to_serve = [m.lower() for m in models_from_env]

logger.info(f"registered models: {registered_models}")
logger.info(f"models to serve: {models_to_serve}")

if models_to_serve and len(models_to_serve) < 1:
    logger.error("models not specified")
    exit(1)

# validate the requested models, if not in registered_models, exit.
for model in models_to_serve:
    if model not in registered_models:
        logger.error("unknown model: " + model)
        exit(1)


logger.info(
    "models: %s, models_dir: %s, gpu_enabled: %s"
    % (models_to_serve, models_base_dir, gpu_enabled),
)
app = Flask(__name__)
api = Api(app)


@api.representation("application/msgpack")
def output_msgpack(data, code, headers=None):
    """Makes a Flask response with a msgpack encoded body"""
    resp = make_response(msgpack.packb(data, use_bin_type=True), code)
    resp.headers.extend(headers or {})
    return resp


# Model directories
base_models_dir = "/models" if not models_base_dir else models_base_dir
roberta_models_dir = os.path.join(base_models_dir, "roberta")
bert_models_dir = os.path.join(base_models_dir, "bert_models")
albert_xlarge_v1_qnli_models_dir = os.path.join(
    base_models_dir, "albert-xlarge-v1-qnli",
)
albert_xxlarge_v1_qnli_models_dir = os.path.join(
    base_models_dir, "albert-xxlarge-v1-qnli",
)
albert_xxlarge_v1_squad2_models_dir = os.path.join(
    base_models_dir, "albert-xxlarge-v1-squad-v2",
)
albert_xlarge_v2_squad2_models_dir = os.path.join(
    base_models_dir, "albert-xlarge-v2-squad-v2",
)
bert_large_wwm_squad2_models_dir = os.path.join(
    base_models_dir, "bert-large-wwm-squad-v2",
)
distilbert_encoder_models_dir = os.path.join(
    base_models_dir, "distilbert-base-nli-stsb-mean-tokens",
)
sif_encoder_models_dir = os.path.join(base_models_dir, "sif")

# chkpoint_file = "model.pt"

# spacy_nlp = spacy.load('en_core_web_lg')
# spacy_nlp = None


def prepare_gpu_model(model, model_name):
    # push model to gpu memory if possible
    if gpu_enabled and torch.cuda.is_available():
        logger.info(f"Using GPU for {model_name}")
        model.cuda()
    model.eval()


# Models
if "qnli" in models_to_serve:
    qnli_model_dir = os.path.join(roberta_models_dir, "roberta.large.qnli")
    qnli_model = RobertaModel.from_pretrained(
        qnli_model_dir,
        gpt2_encoder_json=f"{roberta_models_dir}/encoder.json",
        gpt2_vocab_bpe=f"{roberta_models_dir}/vocab.bpe",
    )
    prepare_gpu_model(qnli_model, "qnli")
    api.add_resource(
        roberta.RobertaQNLI, "/roberta/qnli", resource_class_args=[qnli_model],
    )


if "mnli" in models_to_serve:
    mnli_model_dir = os.path.join(roberta_models_dir, "roberta.large.mnli")
    mnli_model = RobertaModel.from_pretrained(
        mnli_model_dir,
        gpt2_encoder_json=f"{roberta_models_dir}/encoder.json",
        gpt2_vocab_bpe=f"{roberta_models_dir}/vocab.bpe",
    )
    prepare_gpu_model(mnli_model, "mnli")
    api.add_resource(
        roberta.RobertaMNLI, "/roberta/mnli", resource_class_args=[mnli_model],
    )

if "sts-b" in models_to_serve:
    stsb_model_dir = os.path.join(roberta_models_dir, "roberta.large.stsb")
    stsb_model = RobertaModel.from_pretrained(
        stsb_model_dir,
        gpt2_encoder_json=f"{roberta_models_dir}/encoder.json",
        gpt2_vocab_bpe=f"{roberta_models_dir}/vocab.bpe",
    )
    prepare_gpu_model(stsb_model, "sts-b")
    api.add_resource(
        roberta.RobertaSTSB, "/roberta/stsb", resource_class_args=[stsb_model],
    )

if "boolq" in models_to_serve:
    boolq_model_dir = os.path.join(roberta_models_dir, "roberta.large.boolq")
    boolq_model = RobertaModel.from_pretrained(
        boolq_model_dir,
        gpt2_encoder_json=f"{roberta_models_dir}/encoder.json",
        gpt2_vocab_bpe=f"{roberta_models_dir}/vocab.bpe",
    )
    prepare_gpu_model(boolq_model, "boolq")
    api.add_resource(
        roberta.RobertaBOOLQ, "/roberta/boolq", resource_class_args=[boolq_model],
    )

if "squad" in models_to_serve:
    squad_model_dir = os.path.join(bert_models_dir, "squad_output")
    squad_model = transformers.BertForQuestionAnswering.from_pretrained(squad_model_dir)
    bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    api.add_resource(
        bert.BertQA,
        "/bert/squad",
        resource_class_args=["SQUAD", squad_model, bert_tokenizer],
    )

if "albert-xxlarge-v1-squad2" in models_to_serve:
    logger.info("Loading albert-xxlarge-v1-squad2")
    albert_xxlarge_v1_qa_model = transformers.AlbertForQuestionAnswering.from_pretrained(
        albert_xxlarge_v1_squad2_models_dir,
    )
    albert_xxlarge_v1_qa_tokenizer = transformers.AlbertTokenizer.from_pretrained(
        albert_xxlarge_v1_squad2_models_dir,
    )
    prepare_gpu_model(albert_xxlarge_v1_qa_model, "albert xxlarge squad2")
    api.add_resource(
        squad2.BaseQA,
        "/albert-xxlarge-v1/squad2",
        resource_class_args=[
            "ALBERT-xxlarge",
            albert_xxlarge_v1_qa_model,
            albert_xxlarge_v1_qa_tokenizer,
        ],
        endpoint="albert-xxlarge",
    )

if "albert-xlarge-v2-squad2" in models_to_serve:
    logger.info("Loading albert-xlarge-v2-squad2")
    albert_xlarge_qa_model = transformers.AlbertForQuestionAnswering.from_pretrained(
        albert_xlarge_v2_squad2_models_dir,
    )
    albert_xlarge_qa_tokenizer = transformers.AlbertTokenizer.from_pretrained(
        albert_xlarge_v2_squad2_models_dir,
    )
    prepare_gpu_model(albert_xlarge_qa_model, "albert xlarge squad2")
    api.add_resource(
        squad2.BaseQA,
        "/albert-xlarge-v2/squad2",
        resource_class_args=[
            "ALBERT-xlarge",
            albert_xlarge_qa_model,
            albert_xlarge_qa_tokenizer,
        ],
        endpoint="albert-xlarge",
    )

if "albert-xlarge-v1-qnli" in models_to_serve:
    logger.info("Loading albert-xlarge-v1-qnli")
    albert_xlarge_v1_qnli_model = transformers.AlbertForSequenceClassification.from_pretrained(
        albert_xlarge_v1_qnli_models_dir,
    )
    albert_xlarge_v1_qnli_tokenizer = transformers.AlbertTokenizer.from_pretrained(
        albert_xlarge_v1_qnli_models_dir,
    )
    prepare_gpu_model(albert_xlarge_v1_qnli_model, "albert xlarge v1 qnli")
    api.add_resource(
        qnli.BaseQNLI,
        "/albert-xlarge-v1/qnli",
        resource_class_args=[
            "albert-xlarge-v1-qnli",
            albert_xlarge_v1_qnli_model,
            albert_xlarge_v1_qnli_tokenizer,
        ],
        endpoint="albert-xlarge-v1-qnli",
    )


if "albert-xxlarge-v1-qnli" in models_to_serve:
    logger.info("Loading albert-xxlarge-v1-qnli")
    albert_xxlarge_v1_qnli_model = transformers.AlbertForSequenceClassification.from_pretrained(
        albert_xxlarge_v1_qnli_models_dir,
    )
    albert_xxlarge_v1_qnli_tokenizer = transformers.AlbertTokenizer.from_pretrained(
        albert_xxlarge_v1_qnli_models_dir,
    )
    prepare_gpu_model(albert_xxlarge_v1_qnli_model, "albert xxlarge v1 qnli")
    api.add_resource(
        qnli.BaseQNLI,
        "/albert-xxlarge-v1/qnli",
        resource_class_args=[
            "albert-xxlarge-v1-qnli",
            albert_xxlarge_v1_qnli_model,
            albert_xxlarge_v1_qnli_tokenizer,
        ],
        endpoint="albert-xxlarge-v1-qnli",
    )

if "distilbert-encoder" in models_to_serve:
    logger.info("distilbert-encoder")
    distilbert_encoder_model = SentenceTransformer(distilbert_encoder_models_dir)
    prepare_gpu_model(distilbert_encoder_model, "distilbert encoder")
    api.add_resource(
        encoder.DistilBERTEncoder,
        "/distilbert/encoder",
        resource_class_args=["distilbert-encoder", distilbert_encoder_model],
        resource_class_kwargs={
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="distilbert-encoder",
    )


if "sif-encoder" in models_to_serve:
    logger.info("sif-encoder")
    sif_encoder_model = SIFModel(
        f"{sif_encoder_models_dir}/glove.840B.300d.h5",
        f"{sif_encoder_models_dir}/enwiki_vocab_min200.txt",
    )
    api.add_resource(
        encoder.SIFEncoder,
        "/sif/encoder",
        resource_class_args=["sif-encoder", sif_encoder_model],
        resource_class_kwargs={
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="sif-encoder",
    )

# Rest endpoint for healthcheck
api.add_resource(healthcheck.HealthCheck, "/")
logger.info("app initialized")


def main():
    logger.info("Starting app")
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
