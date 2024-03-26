# MODEL Service

Model service is the service layer that hosts a set of smaller NLP task specific models based on RoberTa, Bart, T5 and SIF that provides:

1. Fast inference on cheaper GPUs such as T4 and V100.
2. Easy to fine tune in seconds

This code needs a GPU server to run.

## Getting started

1. Setup MODELS_DIR: Create a models folder on a disk and download required models from Huggingface. To run the following command git-lfs must be enabled.

```
git clone https://huggingface.co/ansukla/roberta
```

The above step requires git-lfs to be installed to pull to large model files. cd to roberta folder, and if the model files are lfs references do the following to pull the models:
```
git lfs fetch
```

```
git clone https://huggingface.co/ansukla/sif
```

The above step requires git-lfs to be installed to pull to large model files. cd to roberta folder, and if the model files are lfs references do the following to pull the models:
```
git lfs fetch
```


2. Download spacy and unzip it such that you have the following folder:

$MODELS_DIR/spacy/en_core_web_lg-3.2.0/ where $MODELS_DIR is the directory you created in 1.

3. After doing step 1 and 2, you should have 3 folders with all the models $MODELS_DIR/roberta, $MODELS_DIR/sif and $MODELS_DIR/spacy

4. Now run the docker image:

Pull the docker image
```
docker pull ghcr.io/nlmatics/nlm-model-service:latest
```
Run the docker image by mapping the port 5001 to port of your choice, mapping your local $MODELS_DIR to /models and specifying a list of models as MODELS environment variable (in csv without commas e.g. MODELS=boolq,sts-b) shown below. The list of supported models are:
 - roberta-qa 
 - boolq
 - roberta-phraseqa
 - sif-encoder
 - nlp_server
 - qa_type
 - sts-b

Although it is possible to run only one model at a time, all the above models must be started to run the nlmatics application.

Other supported models are:
 - bart
 - dpr-encoder
 - flan-t5
 - io_qa
 - qasrl
```
docker run -p 5010:5000 -v /Users/ambikasukla/projects/models/:/models -e MODELS_DIR=/models/ -e MODELS=roberta-qa -e LEARNING_MODELS= nlm-model-service
```
Note that LEARNING_MODELS should always be blank and it is no longer used by the code.

5. Use nlm-utils library to access the model service from your code
```
pip install nlm-utils
```
Examples are available in the nlm-utils repo: https://github.com/nlmatics/nlm-utils

## Credits

This library was created at Nlmatics Corp. from 2019 to 2023.

The scaffolding of this project with initial models and the Phraser was created by Suhail Kandanur. 

Reshav Abraham wrote majority of the nlp pipeline with spacy and added hd5 to speed up glove loading.

Yi Zhang added code for optimal batching and execution on multiple GPUs. He also refactored the code to create a nice class hierarchy and framework for writing new model services. 

Yi Zhang and Nima Sheikloshami contributed the fine tuning code. 

Daniel Ye optimized the SIF serving under guidance of Yi Zhang as documented here: https://blogs.nlmatics.com/nlp/sentence-embeddings/2020/08/07/Smooth-Inverse-Frequency-Frequency-(SIF)-Embeddings-in-Golang.html

Cheyenne Zhang added code for key phrase extaction under guidance of Jasmin Omanovic as documented here: https://blogs.nlmatics.com/keyphrase-extraction/visualization/text/nlp/python/javascript/keppler-mapper/d3/2020/08/07/How-to-Extract-Keyphrases-and-Visualize-Text.html.

Tom Liu added the Yolo code. 

Neima Sheikloshami added the roberta io head code and qasrl code and bio-ner.

Ambika Sukla added code for the roberta qa answer start, end heads. He also added code for batching and similarity calculation with embedding models, optimized the model tuning code and added DPR, T-5 and BART services.

