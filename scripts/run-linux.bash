#!/bin/bash

export FLASK_DEBUG=1
export FLASK_ENV=production
export GPU_ENABLED=0
export MODELS_DIR=/models
export MODELS=qnli
export GPU_ENABLED=false
exec /home/suhail/miniconda3/envs/nlm-model-serving/bin/python app.py $*
