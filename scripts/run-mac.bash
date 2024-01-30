#!/bin/bash

export FLASK_DEBUG=1
export FLASK_ENV=production
export GPU_ENABLED=0
export MODELS_DIR=/Users/suhail/data/nlmatics/data/models
export MODELS=qnli
exec /Users/suhail/miniconda3/envs/nlm-model-serving/bin/python app.py $*
