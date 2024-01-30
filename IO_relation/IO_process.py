import json
from transformers import RobertaTokenizer

def process_squad(dir_):
    
    with open( dir_ ) as file:
        squad_raw = json.load(file)

        squad_raw_data = squad_raw['data']


    input0 = []
    input1 = []
    answers = [] 
    for data in squad_raw_data:
        for par in data['paragraphs']:
            for question in par['qas']:

                input0.append(questions['questions'])
                input1 .append(par['context'])
                answers.append()
                
    return queries, passages



queries_dev, passages_dev = process_squad("/home/ubuntu/nlm/nima.sheikholeslami/Data/Squad/dev-v2.0.json")
queries_train, passages_train = process_squad("/home/ubuntu/nlm/nima.sheikholeslami/Data/Squad/train-v2.0.json")

Tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

