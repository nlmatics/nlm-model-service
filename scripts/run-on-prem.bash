#!/bin/zsh

export FLASK_DEBUG=0
export FLASK_ENV=production
export MODELS_DIR=/Users/ambikasukla/projects/models/
# export MODELS_DIR=/models/models/
export LEARNING_MODELS=

export ACTIVE_LEARNING=true
export CUDA_VISIBLE_DEVICES=0,1

# export MODELS="roberta-qa boolq" #dpr
# export MODELS="bio_ner
# export MODELS="roberta-qa boolq"
# export MODELS="dpr"
# export MODELS="sif-encoder"
# export MODELS="dpr"
export MODELS="sif-encoder"
#export MODELS="boolq qnli sts-b roberta-qa roberta-phraseqa sif-encoder dpr-encoder qa_type nlp_server roberta-encoder"
# export MODELS="boolq"


gunicorn -b 0.0.0.0:5001 --timeout 3600 modelserver_restserver:app
# gunicorn -c /home/ubuntu/ambika/nlm-model-server/scripts/gunicorn_conf.py modelserver_restserver:app

# python3 -m modelserver_restserver