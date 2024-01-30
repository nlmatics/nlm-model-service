#!/bin/bash

export FLASK_DEBUG=0
export FLASK_ENV=production
export MODELS_DIR=/data/models

export CUDA_VISIBLE_DEVICES=0

export MODELS="qnli roberta-qa"
export MODELS="qnli boolq roberta-qa roberta-phraseqa sts-b sif"

export MODELS="qnli roberta-phraseqa roberta-qa boolq sts-b sif-encoder dpr-encoder roberta-encoder"
export MODELS="sif-encoder dpr-encoder roberta-encoder"

gunicorn -b 0.0.0.0:5050 --timeout 3600 test_new_app:app
