#!/bin/bash

export FLASK_ENV=production

nohup /home/suhail/miniconda3/envs/model-serving/bin/python app.py $* 2>&1 1>/tmp/model-serving.log &
