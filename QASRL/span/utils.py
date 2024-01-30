"""
json data:
{'sent': 'There are four boat clubs that row on the River Dee : Aberdeen Boat Club , Aberdeen Schools Rowing Association , Aberdeen University Boat Club and Robert Gordon University Boat Club .',
 'verbs': ['row',
  ['what', 'where'],
  [['four boat clubs',
    'Aberdeen Boat Club',
    'Aberdeen Schools Rowing Association',
    'Aberdeen University Boat Club',
    'Robert Gordon University Boat Club'],
   ['on the River Dee']]]}
"""
############### multi-head multi-span model ###################

import numpy as np


def IO_data_label(Data, tokenizer_func):
    #list of 'question phrases' obtained from the data set
    wh = ['how', 'why', 'when', 'how much', 'what', 'where', 'who']
    out_put = []
    for k, data in enumerate(Data):

        sent = data['sent']
        
        for verb in data['verbs']:

            tokens = tokenizer_func(sent, verb[0])
            wh_ids = np.zeros(len(wh))
                
            question_heads = np.zeros( [len(wh), len(tokens)] )
                
            for i, wh_ in enumerate(verb[1]):
                    wh_idx = wh.index( wh_ )
                    wh_ids[wh_idx] = 1
                    
                    for ans in verb[2][i]:
                        
                        answer_tokens = tokenizer_func(ans)
                        start_char = sent.find(ans)
                        start_token_idx = len(tokenizer_func(sent[:start_char])) - 1
                        end_token_idx = start_token_idx + len(answer_tokens)
                        question_heads[i][ [start_token_idx , end_token_idx] ] = 1
            
            out_put.append((tokens, question_heads, wh_ids))
            
    return out_put                

def BIO_data_label(Data, tokenizer_func):
    """
    Begin 2
    In    1
    Out   0
    BIO
    """

    #list of 'question phrases' obtained from the data set
    wh = ['how', 'why', 'when', 'how much', 'what', 'where', 'who']
    out_put = []
    for k, data in enumerate(Data):

        sent = data['sent']
        
        for verb in data['verbs']:

            tokens = tokenizer_func(sent, verb[0])
            wh_ids = np.zeros(len(wh))
                
            question_heads = np.zeros( [len(wh), len(tokens)] )
                
            for i, wh_ in enumerate(verb[1]):
                    wh_idx = wh.index( wh_ )
                    wh_ids[wh_idx] = 1
                    
                    for ans in verb[2][i]:
                        
                        answer_tokens = tokenizer_func(ans)
                        start_char = sent.find(ans)
                        start_token_idx = len(tokenizer_func(sent[:start_char])) - 1
                        end_token_idx = start_token_idx + len(answer_tokens)
                        question_heads[i][ start_token_idx : end_token_idx ] = 1
                        question_heads[i][start_token_idx] = 2


            out_put.append((tokens, question_heads, wh_ids))
            
    return out_put                


