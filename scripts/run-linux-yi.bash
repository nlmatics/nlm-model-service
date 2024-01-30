#!/bin/bash

export FLASK_DEBUG=0
export FLASK_ENV=production
export MODELS_DIR=/home/yi.zhang/models

export CUDA_VISIBLE_DEVICES=0,1
export N_GPU=1

export ACTIVE_LEARNING=true
#export MODELS="qnli roberta-qa dpr-encoder"
export MODELS="roberta-calc"
export LEARNING_MODELS=""

gunicorn -b 0.0.0.0:5050 --timeout 3600 modelserver_restserver:app
