from distutils.log import debug
import logging
import os
from glob import glob

import msgpack
import torch
from flask import Flask
from flask import make_response
from flask_restful import Api
from transformers import RobertaModel

from models import encoder
from models import healthcheck
from utils.model_utils import get_active_learning_checkpoint

# from models.cross_encoder import CrossEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# disable logging from third-party libraries
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("spacy").setLevel(logging.ERROR)

# check for model folders
models_base_dir = os.environ["MODELS_DIR"]

assert models_base_dir, ValueError("Please set MODELS_DIR to your environment.")

logger.info(f"model dir is set to: {models_base_dir}")

if not os.path.exists(models_base_dir):
    logger.error(f"models dir {models_base_dir} does not exists")
    exit(1)


# check for models
models_from_env = os.environ["MODELS"].split()

assert models_from_env, ValueError(
    "No model to serve, please set MODELS to your environment.",
)
registered_models = [
    "io_qa", 
    "cross_encoder",
    "qasrl",
    "qnli",
    "mnli",
    "sts-b",
    "boolq",
    "squad",
    "roberta-qa",
    "bart",
    "roberta-phraseqa",
    "roberta-calc",
    "roberta-encoder",
    "sif-encoder",
    "dpr-encoder",
    "qa_type",
    "nlp_server",
    "yolo",
    "iorelation",
    "t5",
    "flan-t5",
    "dpr",
    "bio_ner"
]
logger.info(f"registered models: {registered_models}")

learning_models = os.environ["LEARNING_MODELS"].split()

models_to_serve = [m.lower() for m in models_from_env]

# validate the requested models, if not in registered_models, exit.
for model in models_to_serve:
    if model not in registered_models:
        logger.error("unknown model: " + model)
        exit(1)

logger.info(f"models to serve: {models_to_serve}")


logger.info(f"models: {models_to_serve}")
logger.info(f"models_dir: {models_base_dir}")

# start Flask app
app = Flask(__name__)

api = Api(app)


# for msgpack outputs
@api.representation("application/msgpack")
def output_msgpack(data, code, headers=None):
    """Makes a Flask response with a msgpack encoded body"""
    resp = make_response(msgpack.packb(data, use_bin_type=True), code)
    resp.headers.extend(headers or {})
    return resp


def ensure_gpu():
    N_GPU = torch.cuda.device_count()
    assert N_GPU > 0, RuntimeError("Model server requires GPU to run")
    if "N_GPU" in os.environ:
        N_GPU = int(os.getenv("N_GPU"))
    logger.info(f"MODEL-SERVER is running with {N_GPU} GPUs")
    return N_GPU


worker_num = ensure_gpu()


if torch.cuda.is_available():  
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# get flag for active_learning
active_learning = os.getenv("ACTIVE_LEARNING", "true").lower() in {"true", "1"}


logger.info(f"Active Learning: {active_learning} on {worker_num} GPUs")

# Model directories
base_models_dir = "/models" if not models_base_dir else models_base_dir
roberta_models_dir = os.path.join(base_models_dir, "roberta")
gpt2_encoder_json = f"{roberta_models_dir}/encoder.json"
gpt2_vocab_bpe = f"{roberta_models_dir}/vocab.bpe"

sif_encoder_models_dir = os.path.join(base_models_dir, "sif")
dpr_encoder_models_dir = os.path.join(base_models_dir, "dpr")
distilbert_encoder_models_dir = os.path.join(
    base_models_dir,
    "distilbert/sbert.net_models_distilroberta-base-msmarco-v1",
)

if "yolo" in models_to_serve:
    logger.info("Loading YoloV5...")

    try:
        from yolov5.detect import Inference, load_model
    except ImportError as e:
        raise ImportError(f"Cannot load yolo model, {e}")

    model_path = f"{base_models_dir}/yolo/nlm-yolo-r1.pt"
    scaler_path = f"{base_models_dir}/yolo/nlm-scaler-r1.pickle"
    model, scaler, device, names, versions = load_model(model_path, scaler_path)
    imgz = 1344
    api.add_resource(
        Inference,
        "/yolo",
        resource_class_args=[model, scaler, device, names, versions, imgz],
    )
    logger.info("yolo model loaded")

if "qasrl" in models_to_serve:
    try:
        import spacy

        try:
            # try load spacy v3 first
            spacy_nlp = spacy.load(f"{base_models_dir}/spacy/en_core_web_lg-3.2.0/")
            logger.info("spacy v3 loaded")
        except Exception:
            logger.error("can not load spacy v3, trying v2")
            # backward with spacy v2 for MS
            spacy_nlp = spacy.load(f"{base_models_dir}/spacy/en_core_web_lg-2.1.0/")
            logger.info("spacy v2 loaded")

    except ImportError:
        if "NLP_SERVER_HOST" not in os.environ:
            logger.error(
                "spacy model not exist, need to set NLP_SERVER_HOST for spacy functions",
            )
            exit(1)
    from QASRL.span import span_utils
    from models.qasrl import QASRL

    qasrl_model = span_utils.load_span_model(
        f"{base_models_dir}/roberta/roberta.large.qasrl",
        checkpoint="./model.pt",
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )

    api.add_resource(
        QASRL,
        "/roberta/qasrl",
        resource_class_args={
            spacy_nlp,
            qasrl_model,
        },
        endpoint="roberta-qasrl",
    )
    logger.info("qasrl loaded")

if "boolq" in models_to_serve or "nlp_server" in models_to_serve:
    try:
        import spacy

        try:
            # try load spacy v3 first
            spacy_nlp = spacy.load(f"{base_models_dir}/spacy/en_core_web_lg-3.2.0/")
            logger.info("spacy v3 loaded")
        except Exception:
            logger.error("can not load spacy v3, trying v2")
            # backward with spacy v2 for MS
            spacy_nlp = spacy.load(f"{base_models_dir}/spacy/en_core_web_lg-2.1.0/")
            logger.info("spacy v2 loaded")

    except ImportError:
        if "NLP_SERVER_HOST" not in os.environ:
            logger.error(
                "spacy model not exist, need to set NLP_SERVER_HOST for spacy functions",
            )
            exit(1)
        spacy_nlp = None


if "cross_encoder" in models_to_serve:
    logger.info("Loading cross_encoder...")
    logger.info(f"{base_models_dir}")
    from managed_models.Transformers import CrossEncoderResource
    from managed_models.Transformers import CrossEncoderManager

    Cross_Encoder_manager = CrossEncoderManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.crossencoder/",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.crossencoder/",
            "model.pt",
        ),
        worker_num=worker_num,
        active_learning=active_learning and "cross_encoder" in learning_models,
    )

    api.add_resource(
        CrossEncoderResource,
        "/roberta/CrossEncoder",
        resource_class_kwargs={
            "model_manager": Cross_Encoder_manager,
        },
        endpoint="cross-encoder",
    )
    logger.info("cross_encoder loaded!!")

if "bio_ner" in models_to_serve:
    logger.info("Loading bioner models ...")
    logger.info(f"{base_models_dir}")    
    from managed_models.Transformers import BioNERResource
    from transformers import CanineTokenizer, CanineForTokenClassification
    import torch

    cdr_model = None
    cdr_tokenizer = None

    cdr_tokenizer = CanineTokenizer.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/cdr-model", num_labels=3)
    cdr_model = CanineForTokenClassification.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/cdr-model", num_labels=3)
    cdr_model.to(device)

    # mutation_tokenizer = None
    # mutation_model = None
    mutation_tokenizer = CanineTokenizer.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/mutation-model", num_labels=2)
    mutation_model = CanineForTokenClassification.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/mutation-model", num_labels=2)
    mutation_model.to(device)

    gene_tokenizer = CanineTokenizer.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/gene-model", num_labels=2)
    gene_model = CanineForTokenClassification.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/gene-model", num_labels=2)
    gene_model.to(device)

    cell_tokenizer = CanineTokenizer.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/cell-model", num_labels=2)
    cell_model = CanineForTokenClassification.from_pretrained(f"{models_base_dir}/huggingface/bio-ner/cell-model", num_labels=2)
    cell_model.to(device)

    api.add_resource(
        BioNERResource,
        "/bio_ner/tag",
        resource_class_kwargs={
        "cdr_tokenizer": cdr_tokenizer, 
        "cdr_model": cdr_model,
        "mutation_tokenizer": mutation_tokenizer,
        "mutation_model": mutation_model,
        "gene_tokenizer": gene_tokenizer,
        "gene_model": gene_model,
        "cell_model": cell_model,
        "cell_tokenizer": cell_tokenizer
        },
        endpoint="tag",
    )
    logger.info("bio_ner loaded!!")

if "t5" in models_to_serve:
    logger.info("Loading t5-large ...")
    logger.info(f"{base_models_dir}")    
    from managed_models.Transformers import T5Resource
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained(f"{models_base_dir}/huggingface/t5-xl")
    model = T5ForConditionalGeneration.from_pretrained(f"{models_base_dir}/huggingface/t5-xl").to(device)

    api.add_resource(
        T5Resource,
        "/t5/infer",
        resource_class_kwargs={"tokenizer": tokenizer, "model": model},
        endpoint="infer",
    )
    logger.info("T5 loaded!!")

if "flan-t5" in models_to_serve:
    logger.info("Loading flan-t5 model ...")
    logger.info(f"{base_models_dir}")    
    from managed_models.Transformers import FlanT5Resource
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained(f"{models_base_dir}/huggingface/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained(f"{models_base_dir}/huggingface/flan-t5-large").to(device)
    model.cuda().eval()
    api.add_resource(
        FlanT5Resource,
        "/flan-t5/infer",
        resource_class_kwargs={"tokenizer": tokenizer, "model": model},
        endpoint="infer",
    )
    logger.info("Flan T5 loaded!!")

if "bart" in models_to_serve:
    logger.info("Loading bart model ...")
    logger.info(f"{base_models_dir}")    
    from managed_models.Transformers import BartResource
    from transformers import BartTokenizerFast, BartForConditionalGeneration
    bart_path = f"{models_base_dir}/huggingface/task-llm"
    # bart_path = '/home/ubuntu/ambika/models/huggingface/task-llm'
    tokenizer = BartTokenizerFast.from_pretrained(bart_path)
    model = BartForConditionalGeneration.from_pretrained(bart_path).to(device)
    model.cuda().eval()
    api.add_resource(
        BartResource,
        "/bart/infer",
        resource_class_kwargs={"tokenizer": tokenizer, "model": model},
        endpoint="infer",
    )
    logger.info("BART loaded!!")

if "dpr" in models_to_serve:
    logger.info("Loading dpr ...")
    logger.info(f"{base_models_dir}")    
    from managed_models.Transformers import DPRResource
    from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoder
    context_encoder = DPRContextEncoder.from_pretrained(f'{base_models_dir}/huggingface/dpr-ctx_encoder-single-nq-base').to(device)
    question_encoder = DPRQuestionEncoder.from_pretrained(f'{base_models_dir}/huggingface/dpr-question_encoder-single-nq-base').to(device)
    question_encoder.cuda().eval()
    context_encoder.cuda().eval()
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f'{base_models_dir}/huggingface/dpr-ctx_encoder-single-nq-base')
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f'{base_models_dir}/huggingface/dpr-question_encoder-single-nq-base')
    api.add_resource(
        DPRResource,
        "/dpr-question/encoder",
        resource_class_kwargs={
            "tokenizer": question_tokenizer, 
            "encoder": question_encoder,
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="encode-question",
    )
    api.add_resource(
        DPRResource,
        "/dpr-context/encoder",
        resource_class_kwargs={
            "tokenizer": question_tokenizer, 
            "encoder": context_encoder,
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="encode-context",
    )
    logger.info("dpr loaded!!")


if "io_qa" in models_to_serve:
    logger.info("Loading io_qa ...")
    logger.info(f"{base_models_dir}")
    from managed_models.Transformers import IOQAResource
    from managed_models.Transformers import IOQAManager

    Cross_Encoder_manager = IOQAManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.io_qa/",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.IO_QA/",
            "model.pt",
        ),
        worker_num=worker_num,
        active_learning=active_learning and "io_qa" in learning_models,
    )

    api.add_resource(
        IOQAResource,
        "/roberta/io-qa",
        resource_class_kwargs={
            "model_manager": Cross_Encoder_manager,
        },
        endpoint="io-qa",
    )
    logger.info("IO_QA loaded!!")





if "iorelation" in models_to_serve:
    logger.info("Loading IORelation ...")
    logger.info(f"{base_models_dir}")
    from managed_models.Transformers import TransformerIORelationResource
    from managed_models.Transformers import TransformerIORelationManager

    IOQA_manager = TransformerIORelationManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.IORelation/",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.IORelation/",
            "model.pt",
        ),
        worker_num=worker_num,
        active_learning=active_learning and "IORelation" in learning_models,
    )

    api.add_resource(
        TransformerIORelationResource,
        "/roberta/IORelation",
        resource_class_kwargs={
            "model_manager": IOQA_manager,
        },
        endpoint="IORelation",
    )
    logger.info("IORelation loaded!!")

if "qnli" in models_to_serve:
    logger.info("Loading qnli...")
    from managed_models.roberta import RobertaQNLIResource
    from managed_models.roberta import RobertaQNLIManager

    qnli_manager = RobertaQNLIManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.qnli",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.qnli",
            "model.pt",
        ),
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
        worker_num=worker_num,
        active_learning=active_learning and "qnli" in learning_models,
    )

    api.add_resource(
        RobertaQNLIResource,
        "/roberta/qnli",
        resource_class_kwargs={
            "model_manager": qnli_manager,
        },
        endpoint="roberta-qnli",
    )
    logger.info("qnli loaded")

if "mnli" in models_to_serve:
    logger.info("Loading mnli...")
    from managed_models.roberta import RobertaMNLIResource
    from managed_models.roberta import RobertaMNLIManager

    mnli_manager = RobertaMNLIManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.mnli",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.mnli",
            "model.pt",
        ),
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
        worker_num=worker_num,
        active_learning=active_learning and "mnli" in learning_models,
    )

    api.add_resource(
        RobertaMNLIResource,
        "/roberta/mnli",
        resource_class_kwargs={
            "model_manager": mnli_manager,
        },
        endpoint="roberta-mnli",
    )
    logger.info("mnli loaded")

if "sts-b" in models_to_serve:
    logger.info("Loading stsb...")
    from managed_models.roberta import RobertaSTSBResource
    from fairseq.models.roberta.model import RobertaModel

    stsb_model = RobertaModel.from_pretrained(
        f"{base_models_dir}/roberta/roberta.large.stsb",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.stsb",
            "model.pt",
        ),
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    stsb_model.half()
    stsb_model.cuda()
    stsb_model.eval()
    api.add_resource(
        RobertaSTSBResource,
        "/roberta/stsb",
        resource_class_kwargs={
            "head": "sentence_classification_head",
            "model": stsb_model,
        },
        endpoint="roberta-stsb",
    )
    logger.info("roberta stsb loaded")

if "qa_type" in models_to_serve:
    logger.info("Loading qa_type...")
    from managed_models.roberta import RobertaQATypeResource
    from fairseq.models.roberta.model import RobertaModel

    qa_type_model = RobertaModel.from_pretrained(
        f"{base_models_dir}/roberta/roberta.large.qatype.lower.RothWithQ",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.qatype.lower.RothWithQ",
            "model.pt",
        ),
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    qa_type_model.half()
    qa_type_model.cuda()
    qa_type_model.eval()
    logger.info("roberta qa-type loaded")

    api.add_resource(
        RobertaQATypeResource,
        "/roberta/qa_type",
        resource_class_kwargs={
            "head": "sentence_classification_head",
            "model": qa_type_model,
        },
        endpoint="roberta-qa_type",
    )
    logger.info("qa_type loaded")


if "boolq" in models_to_serve:
    from managed_models.roberta import RobertaBOOLQResource
    from fairseq.models.roberta.model import RobertaModel
    model_dir = f"{base_models_dir}/roberta/roberta.large.boolq"
    checkpoint_file=get_active_learning_checkpoint(
        model_dir,
        "model.pt",
    )
    logger.info(f"Loading boolq model: {checkpoint_file}")

    boolq_model = RobertaModel.from_pretrained(
        model_dir,
        checkpoint_file,
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    boolq_model.half()
    boolq_model.cuda()
    boolq_model.eval()
    logger.info("roberta boolq loaded")

    api.add_resource(
        RobertaBOOLQResource,
        "/roberta/boolq",
        resource_class_kwargs={
            "spacy_nlp": spacy_nlp,
            "head": "sentence_classification_head",
            "model": boolq_model,
            "model_dir": model_dir 
        },
        endpoint="roberta-boolq",
    )
    logger.info("boolq loaded")

if "roberta-qa" in models_to_serve:
    logger.info("Loading roberta qa...")
    from managed_models.roberta import RobertaQAResource
    from roberta.span.span_model import SpanModel
    roberta_qa_model_dir = f"{roberta_models_dir}/roberta.large.qa"
    checkpoint_file=get_active_learning_checkpoint(
        roberta_qa_model_dir,
        "model.pt",
    ),
    checkpoint_file = checkpoint_file[0]
    logger.info(f"Loading roberta qa model: {checkpoint_file}")
    roberta_qa_model = SpanModel.from_pretrained(
        roberta_qa_model_dir,
        checkpoint_file=checkpoint_file,
        arch="roberta_span_large",
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    roberta_qa_model.half()
    roberta_qa_model.to(device)
    roberta_qa_model.cuda()
    roberta_qa_model.eval()
    logger.info("roberta qa loaded")
    api.add_resource(
        RobertaQAResource,
        "/roberta/qa",
        resource_class_kwargs={
            "model": roberta_qa_model,
            "head": "span",
            "model_dir": roberta_qa_model_dir
        },
        endpoint="qa",
    )
    # backward compability
    api.add_resource(
        RobertaQAResource,
        "/roberta/roberta-qa",
        resource_class_kwargs={
            "model": roberta_qa_model,
            "head": "span",
            "model_dir": roberta_qa_model_dir
        },
        endpoint="roberta-qa",
    )
    logger.info("roberta qa resource ready")


if "roberta-phraseqa" in models_to_serve:
    from managed_models.roberta import RobertaPhraseQAResource
    from roberta.span.span_model import SpanModel

    logger.info("Loading roberta phrase qa...")
    roberta_phrase_qa_model_dir = f"{roberta_models_dir}/roberta.large.phraseqa"
    checkpoint_file=get_active_learning_checkpoint(
        roberta_phrase_qa_model_dir,
        "model.pt",
    ),
    checkpoint_file = checkpoint_file[0]
    logger.info(f"Loading roberta qa model: {checkpoint_file}")

    roberta_phraseqa_model = SpanModel.from_pretrained(
        roberta_phrase_qa_model_dir,
        checkpoint_file=checkpoint_file,
        arch="roberta_span_large",
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
    )
    roberta_phraseqa_model.half()
    roberta_phraseqa_model.to(device)
    roberta_phraseqa_model.cuda()
    roberta_phraseqa_model.eval()
    logger.info("roberta phrase qa loaded")

    api.add_resource(
        RobertaPhraseQAResource,
        "/roberta/phraseqa",
        resource_class_kwargs={
            "model": roberta_phraseqa_model,
            "head": "span",
            "model_dir": roberta_models_dir
        },
        endpoint="phrase-qa",
    )

    # backward compability
    api.add_resource(
        RobertaPhraseQAResource,
        "/roberta/roberta-phraseqa",
        resource_class_kwargs={
            "model": roberta_phraseqa_model,
            "head": "span",
            "model_dir": roberta_models_dir
        },
        endpoint="roberta-phrase-qa",
    )

    logger.info("phrase qa loaded")

if "roberta-calc" in models_to_serve:
    logger.info("Loading roberta calc...")
    from managed_models.roberta import RobertaCalcResource
    from managed_models.roberta import RobertaCalcManager

    calc_manager = RobertaCalcManager.get_manager(
        f"{base_models_dir}/roberta/roberta.large.calc",
        checkpoint_file=get_active_learning_checkpoint(
            f"{base_models_dir}/roberta/roberta.large.calc",
            "model.pt",
        ),
        gpt2_encoder_json=gpt2_encoder_json,
        gpt2_vocab_bpe=gpt2_vocab_bpe,
        worker_num=worker_num,
        active_learning=active_learning and "roberta-calc" in learning_models,
    )

    api.add_resource(
        RobertaCalcResource,
        "/roberta/calc",
        resource_class_kwargs={
            "model_manager": calc_manager,
        },
        endpoint="calc",
    )
    # backward compability
    api.add_resource(
        RobertaCalcResource,
        "/roberta/roberta-calc",
        resource_class_kwargs={
            "model_manager": calc_manager,
        },
        endpoint="roberta-calc",
    )
    logger.info("calc loaded")

if "sif-encoder" in models_to_serve:

    try:
        from sif.cython_sif import SIFModel
    except ImportError as e:
        logger.error(
            f"cython sif is not found, fall back to python implementation. {e}",
        )
        from sif.sif import SIFModel

    logger.info("loading sif-encoder...")
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
    api.add_resource(
        encoder.SIFSimilarity,
        "/sif/similarity",
        resource_class_args=["similarity", sif_encoder_model],
        resource_class_kwargs={
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="similarity",
    )

    logger.info("sif-encoder loaded")


if "dpr-encoder" in models_to_serve:
    logger.info("loading dpr-encoder...")
    try:
        from dpr.dpr import get_dpr_model
    except ImportError as e:
        raise ImportError(f"Can not load DPR model, {e}")

    context_encoder, question_encoder = get_dpr_model(
        dpr_encoder_models_dir,
        "dpr_biencoder.pt",
    )

    context_encoder = context_encoder.half().cuda().eval()
    question_encoder = question_encoder.half().cuda().eval()

    api.add_resource(
        encoder.DPREncoder,
        "/dpr-question/encoder",
        resource_class_args=["dpr-question", question_encoder],
        resource_class_kwargs={
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="dpr-question",
    )

    api.add_resource(
        encoder.DPREncoder,
        "/dpr-context/encoder",
        resource_class_args=["dpr-context", context_encoder],
        resource_class_kwargs={
            "representations": {"application/msgpack": output_msgpack},
        },
        endpoint="dpr-context",
    )

    logger.info("dpr-encoder loaded")

if "roberta-encoder" in models_to_serve:
    logger.info("Loading roberta-encoder...")
    model_dir = os.path.join(roberta_models_dir, "roberta.large.stsb")

    def prepare_roberta_encoder_model(model_dir):

        # load model for encoding
        model = RobertaModel.from_pretrained(
            model_dir,
            gpt2_encoder_json=f"{roberta_models_dir}/encoder.json",
            gpt2_vocab_bpe=f"{roberta_models_dir}/vocab.bpe",
        )
        model = model.half()
        model = model.cuda()
        model = model.eval()

        return model

    api.add_resource(
        encoder.RobertaEncoder,
        "/roberta/encoder",
        resource_class_args=[prepare_roberta_encoder_model(model_dir)],
    )
    logger.info("roberta-encoder loaded")

if "nlp_server" in models_to_serve:
    if spacy_nlp is None:
        raise ImportError("can not load spacy model")
        
    try:
        import spacy
        from nlp_server.phraser import Phraser
        bio_nlp = None
        try:
            # try load spacy v3 first
            bio_nlp = [
                spacy.load(f"{base_models_dir}/spacy/en_ner_craft_md-0.5"),
                spacy.load(f"{base_models_dir}/spacy/en_ner_bc5cdr_md-0.5"),
                spacy.load(f"{base_models_dir}/spacy/en_ner_jnlpba_md-0.5"),
                spacy.load(f"{base_models_dir}/spacy/en_ner_bionlp13cg_md-0.5"),
            ]
            
            logger.info("scispacy for bioNER is loaded")
        except Exception:
            logger.error("can not load BioNER", exc_info=True)
                

        logger.info("Loading nlp_server...")

        api.add_resource(
            Phraser,
            "/nlp",
            resource_class_args=(spacy_nlp,bio_nlp),
        )
        logger.info("nlp_server loaded")
    except ImportError:
        if "NLP_SERVER_HOST" not in os.environ:
            logger.error(
                "spacy model not exist, need to set NLP_SERVER_HOST for spacy functions",
            )
            exit(1)    


# Rest endpoint for healthcheck
api.add_resource(healthcheck.HealthCheck, "/healthz")
logger.info("app initialized")


def main():
    logger.info("Starting app")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
