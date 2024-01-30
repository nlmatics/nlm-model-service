import torch
import numpy as np
import pickle
import json
from fairseq.models.roberta.model import RobertaModel
from transformers import AutoTokenizer
import re


# First comes the sentences and the the realtion is added
Labels_2_sent = {
 'subsidiaries(e1,e2)' : "subsidiary" ,
 'alternate_names(e1,e2)' : "alternate name" ,
 'alternate_names(e2,e1)' : "alternate name",
 'countries_of_residence(e1,e2)': "country of residence" ,
 'founded_by(e1,e2)': "founded by",
 'cities_of_residence(e1,e2)': "city of residence",
 'top_members/employees(e2,e1)': "employee",
 'members(e1,e2)': "member",
 'origin(e1,e2)': "origin",
 'employee_of(e1,e2)': "employee",
 'countries_of_residence(e2,e1)': "county of residence",
 'founded(e1,e2)': "founded",
 'spouse(e2,e1)': "spouse",
 'alternate_names(e2,e1)': "alternate name",
 'founded(e2,e1)': "is founded by",
 'city_of_headquarters(e2,e1)': "city of headquarters",
 'stateorprovinces_of_residence(e2,e1)': "state or province of residence",
 'founded_by(e2,e1)': "founded by",
 'members(e2,e1)': "member",
 'country_of_headquarters(e2,e1)': "country of headquareters",
 'cities_of_residence(e2,e1)': "city of residence" ,
 'title(e2,e1)' : "title",
 'country_of_birth(e1,e2)' : "country of birth of",
 'stateorprovince_of_headquarters(e2,e1)' : "state or province of headquarters",
 'city_of_headquarters(e1,e2)': "city of headquarters",
 'title(e1,e2)': "title",
 'country_of_birth(e2,e1)': "country of birth",
 'top_members/employees(e1,e2)' : "employee of",
 'country_of_headquarters(e1,e2)': "country of headquarters",
 'stateorprovinces_of_residence(e1,e2)': "state or province of residence",
 'stateorprovince_of_headquarters(e1,e2)' : "state or province of headquarters",
 'origin(e2,e1)' : "origin",
 'subsidiaries(e2,e1)' : "subsidiary",
 'employee_of(e2,e1)' : "employee",
 'spouse(e1,e2)': "spouse"
    
}



models_dir = "/home/ubuntu/nlm/nima/Data/Models/boolq_for_rel/"
model = RobertaModel.from_pretrained(
            models_dir,
            "./model.pt",
            head="classification",
            gpt2_encoder_json=f"{models_dir}/encoder.json",
            gpt2_vocab_bpe=f"{models_dir}/vocab.bpe"
)




def find_start_end_positions(sentence, answer, start_char):
    answer_tokens = model.encode(answer)[1:-1]
    start_token_idx = len(model.encode(sentence[:start_char].strip())) -1 
    end_token_idx = start_token_idx + len(answer_tokens)

    return start_token_idx, end_token_idx





def get_ent(sent):
    r1 = re.search("<e1>(.*)</e1>",sent)
    r2 = re.search("<e2>(.*)</e2>",sent)
    return r1.group(1).strip(), r2.group(1).strip()



def clean_sent(sent):
    
    sent = sent.replace("<e1>","")
    sent = sent.replace("</e1>","")
    sent = sent.replace("<e2>","")
    sent = sent.replace("</e2>","")
    sent_ = sent.split(" ")
    try:
        while True:
            sent_.remove("")
    except:
        sent = " ".join(sent_)
        
    return sent



def process_kbp_fairseq(dir_txt):
    
    with open(dir_txt) as file:
        data = file.read()
    data = data.split("\n")[:-1]

    data_sentences, data_labels = [], []
    for i in range(0, len(data), 4):
        data_sentences.append(data[i].split("\t")[1][1:-1].strip())
        data_labels.append(data[i+1].strip())

    Start_positions = []
    End_positions = []
    Tokens = []

    for i, (data_,label_) in enumerate(zip( data_sentences, data_labels)):
        print(i)
        if label_ == 'no_relation':
            continue
        _, relation = label_.split(":")
        relation = relation.strip()


        head, tail = get_ent(data_)
        cln_sentence = clean_sent(data_)
        relation = Labels_2_sent[ relation ]
        head_start = cln_sentence.find(head)
        tail_start = cln_sentence.find(tail)
        head_start_token_idx, head_end_token_idx = find_start_end_positions(cln_sentence, head,head_start)
        tail_start_token_idx, tail_end_token_idx = find_start_end_positions(cln_sentence, tail, tail_start)
        Start_positions.append( (head_start_token_idx, tail_start_token_idx))
        End_positions.append( (head_end_token_idx, tail_end_token_idx))
        Tokens.append(
            model.encode(relation, cln_sentence).tolist()
        )



    
    return Tokens, Start_positions, End_positions




def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    questions = [example['relation'] for example in examples]
    sentences = [example['sentence'] for example in examples]
    inputs = tokenizer(
        questions,
        sentences,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors='pt'
    )

    offset_mapping = inputs.pop("offset_mapping")


    
    
    
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):

        head = examples[i]['head']
        tail = examples[i]['tail']

        try:
            head_start_char = head['start']
            head_end_char = head['start'] + len(head['word'])
        except:
            head_start_char, head_end_char = 0, 0


        try:
            tail_start_char = tail['start']
            tail_end_char = tail['start'] + len(tail['word'])
        except:
            tail_start_char, tail_end_char = 0, 0



        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        #if start_char and end_char are both zero, label it (0, 0)
        if (head_start_char, head_end_char) == (0, 0):
            head_start_position, head_end_position = 0, 0

        # If the answer is not fully inside the context, label it (0, 0)
        elif offset[context_start][0] > head_end_char or offset[context_end][1] < head_start_char:
            head_start_position, head_end_position = 0, 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= head_start_char:
                idx += 1
            head_start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset[idx][1] >= head_end_char:
                idx -= 1
            head_end_position = idx + 2





        if (tail_start_char, tail_end_char) == (0, 0):
            tail_start_position, tail_end_position = 0, 0

        # If the answer is not fully inside the context, label it (0, 0)
        elif offset[context_start][0] > tail_end_char or offset[context_end][1] < tail_start_char:
            tail_start_position, tail_end_position = 0, 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= tail_start_char:
                idx += 1
            tail_start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset[idx][1] >= tail_end_char:
                idx -= 1
            tail_end_position = idx + 2


        start_positions.append( (head_start_position, tail_start_position) )
        end_positions.append( (head_end_position, tail_end_position) )

    return inputs['input_ids'], start_positions, end_positions



def process_kbp_huggingface(dir_txt, save_dir, file_type='train'):
        
    with open(dir_txt) as file:
        data = file.read()
    data = data.split("\n")[:-1]

    data_sentences, data_labels = [], []
    for i in range(0, len(data), 4):
        data_sentences.append(data[i].split("\t")[1][1:-1].strip())
        data_labels.append(data[i+1].strip())

    Start_positions = []
    End_positions = []
    Input_ids= []
    Examples = []
    for i, (data_,label_) in enumerate(zip( data_sentences, data_labels)):
        print(i)
        if label_ == 'no_relation':
            continue
        _, relation = label_.split(":")
        relation = relation.strip()


        head, tail = get_ent(data_)
        cln_sentence = clean_sent(data_)
        relation = Labels_2_sent[ relation ]
        head_start = cln_sentence.find(head)
        tail_start = cln_sentence.find(tail)
        Examples.append(
            {
                'sentence' : cln_sentence, 
                'relation':relation, 
                'head' : {
                    'word':head,
                    'start': head_start
                }, 
                'tail' : {
                    'word' : tail, 
                    'start' : tail_start
                }
            }
        )
    print('started tokenizion')
    for i in range(0, len(Examples), 100):
        print(i)
        input_ids, start_positions, end_positions = preprocess_function(Examples[i:i+100])
        Input_ids.append(input_ids)
        Start_positions.extend(start_positions)
        End_positions.extend(end_positions)
        
    Input_ids = torch.cat(Input_ids, axis =0)
    print("saving data!!")
    torch.save(Input_ids, f"{save_dir}/input_ids_{file_type}.pt")
    np.save(f"{save_dir}/start_positions_{file_type}", Start_positions)
    np.save(f'{save_dir}/end_positions_{file_type}', End_positions)




def main(dir_, save_dir="", file_type='train'):

    Tokens, Start_positions, End_positions = process_kbp_fairseq(dir_)
    
    np.save(save_dir + f"/start_positions_{file_type}", Start_positions )
    np.save(save_dir + f"/end_positions_{file_type}", End_positions )
    
    with open(save_dir + f"not_collated_{file_type}_set_tokens", "wb") as f:
        pickle.dump(Tokens, f)


    #short versions
    

if __name__ == "__main__":

    
    dir_ = '/home/ubuntu/nlm/nima/Data/kbp37/'
    dir_dev = f'{dir_}/train.txt'
    dir_train = f'{dir_}/dev.txt'
    save_dir = f'{dir_}/IO_preprocess_huggingface/'
    process_kbp_huggingface(dir_dev, save_dir, file_type='dev')
    process_kbp_huggingface(dir_train, save_dir, file_type='train')


    """



    print('processing train data')
    dir_ = '/home/ubuntu/nlm/nima/Data/kbp37/'
    dir_train = f'{dir_}/train.txt'
    save_dir = f'{dir_}/IO_preprocess_fair/'
    main(dir_train, save_dir, file_type='train')


    dir_dev = f'{dir_}/dev.txt'
    main(dir_dev, save_dir, file_type='dev')    
    """