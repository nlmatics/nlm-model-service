#!/bin/bash
export FLASK_DEBUG=0
export FLASK_ENV=production
export MODELS_DIR=/models
#export MODELS="qnli albert-xxlarge-v1-qnli squad albert-xxlarge-squad2"
#
#    'albert-xlarge-squad2',
#    'albert-xxlarge-squad2',
#    'bert-large-wwm-squad2',
#    'roberta-base-squad2',


export MODELS="qnli boolq sts-b roberta-qa roberta-phraseqa dpr-encoder sif-encoder"
#python3 app.py $*

gunicorn -b 0.0.0.0:5001 --timeout 3600 modelserver_restserver:app
