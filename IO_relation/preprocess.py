import torch
import numpy as np
import pickle
import json
from fairseq.models.roberta.model import RobertaModel
from transformers import AutoTokenizer


# First comes the sentences and the the realtion is added
label2sent = {
 'NA' : 'Nan', 
 'active_metabolites_of' : "Nan",
 'anatomic_structure_has_location' : 'has',
 'anatomic_structure_is_physical_part_of': 'is physical part of',
 'anatomy_originated_from_biological_process' : 'orginated from',
 'associated_with_malfunction_of_gene_product' : 'is associated with',
 'biological_process_has_associated_location' : 'has',
 'biological_process_has_initiator_chemical_or_drug' : 'has initiator',
 'biological_process_has_initiator_process' : 'has initiator',
 'biological_process_has_result_anatomy' : 'has result',
 'biological_process_has_result_biological_process' : 'has result',
 'biological_process_has_result_chemical_or_drug' : 'has result',
 'biological_process_involves_gene_product' : 'involves',
 'biological_process_is_part_of_process': 'is part of',
 'biological_process_results_from_biological_process' : 'results from',
 'biomarker_type_includes_gene_product' : 'includes',
 'cdrh_parent_of': 'is parent of',
 'chemical_or_drug_affects_gene_product' : 'affects',
 'chemical_or_drug_initiates_biological_process' : 'initiates',
 'chemical_or_drug_is_product_of_biological_process' : 'is product of',
 'chemical_structure_of': " is structure of",
 'chemotherapy_regimen_has_component' : 'has',
 'completely_excised_anatomy_has_procedure' : 'has',
 'complex_has_physical_part' : 'has',
 'concept_in_subset': 'Nan',
 'conceptual_part_of': 'Nan',
 'contraindicated_with_disease' : 'contraindicated with',
 'contraindicating_class_of': 'Nan',
 'disease_excludes_normal_cell_origin' : 'excludes',
 'disease_excludes_primary_anatomic_site' : 'excludes',
 'disease_has_abnormal_cell': 'has',
 'disease_has_associated_anatomic_site' : 'is associated with',
 'disease_has_associated_disease' : 'is associated with',
 'disease_has_associated_gene' : 'is associated with',
 'disease_has_finding' : 'has',
 'disease_has_metastatic_anatomic_site' : 'has',
 'disease_has_normal_cell_origin' : 'has',
 'disease_has_normal_tissue_origin' : 'has',
 'disease_has_primary_anatomic_site' : 'has',
 'disease_may_have_associated_disease' : 'maybe associated with',
 'disease_may_have_finding' : 'Nan',
 'excised_anatomy_has_procedure': 'has',
 'gene_associated_with_disease' : 'is associated with',
 'gene_encodes_gene_product' : 'encodes',
 'gene_found_in_organism' : 'found in',
 'gene_mapped_to_disease' : 'mapped to',
 'gene_plays_role_in_process' : 'plays role in',
 'gene_product_affected_by_chemical_or_drug' : 'affected by',
 'gene_product_encoded_by_gene' : 'encoded by',
 'gene_product_expressed_in_tissue' : 'expressed in',
 'gene_product_has_associated_anatomy' : 'has associated',
 'gene_product_has_biochemical_function' : 'has',
 'gene_product_has_chemical_classification': 'has',
 'gene_product_has_organism_source' : 'has',
 'gene_product_has_structural_domain_or_motif' : 'has',
 'gene_product_is_biomarker_of' : 'is biomaker of',
 'gene_product_is_physical_part_of' : 'is physical part of',
 'gene_product_malfunction_associated_with_disease' : 'is associated with',
 'gene_product_plays_role_in_biological_process': 'playes role in',
 'has_active_metabolites': 'has active',
 'has_cdrh_parent': 'has',
 'has_chemical_structure': 'has',
 'has_conceptual_part': 'Nan',
 'has_contraindicated_drug': 'has contraindicated',
 'has_contraindicating_class': 'Nan',
 'has_free_acid_or_base_form': 'has',
 'has_ingredient': 'has',
 'has_mechanism_of_action': 'has',
 'has_nichd_parent': 'has',
 'has_physical_part_of_anatomic_structure' : 'has',
 'has_physiologic_effect': 'has',
 'has_salt_form' : 'has',
 'has_therapeutic_class' : 'has',
 'has_tradename' : 'has',
 'induced_by' : 'is induced by',
 'induces' : 'induces',
 'ingredient_of' : 'is an ingredient of',
 'is_abnormal_cell_of_disease' : 'is abnormal cell of',
 'is_associated_anatomic_site_of' : 'is associated with',
 'is_associated_anatomy_of_gene_product' : 'is associated with',
 'is_associated_disease_of' : 'is associated with',
 'is_biochemical_function_of_gene_product' : 'Nan',
 'is_chemical_classification_of_gene_product' : 'Nan',
 'is_component_of_chemotherapy_regimen' : 'is component of',
 'is_finding_of_disease' : 'Nan',
 'is_location_of_anatomic_structure' : 'is location of',
 'is_location_of_biological_process' : 'is location of',
 'is_marked_by_gene_product' : "is marked by",
 'is_metastatic_anatomic_site_of_disease' : 'Nan',
 'is_normal_cell_origin_of_disease' : 'Nan',
 'is_normal_tissue_origin_of_disease' : 'Nan',
 'is_not_normal_cell_origin_of_disease' : 'Nan',
 'is_not_primary_anatomic_site_of_disease' : 'is not primary anotomic site of',
 'is_organism_source_of_gene_product' : 'is organism source of',
 'is_physiologic_effect_of_chemical_or_drug' : 'is physiologic effect of',
 'is_primary_anatomic_site_of_disease' : 'is primary anatomic site of',
 'is_structural_domain_or_motif_of_gene_product': 'is structural domain or motif of',
 'may_be_associated_disease_of_disease' : 'may be associated disease of',
 'may_be_diagnosed_by' : 'may be diagnosed by',
 'may_be_finding_of_disease' : 'may be finding of',
 'may_be_prevented_by' : 'may be prevented by',
 'may_be_treated_by' : 'may be treated by',
 'may_diagnose' : 'may diagnose',
 'may_prevent' : 'may prevent',
 'may_treat' : 'may treat',
 'mechanism_of_action_of' : 'Nan',
 'nichd_parent_of' : 'Nan',
 'organism_has_gene' : 'has',
 'partially_excised_anatomy_has_procedure' : 'has',
 'pathogenesis_of_disease_involves_gene' : 'involves',
 'physiologic_effect_of' : 'is effect of',
 'procedure_has_completely_excised_anatomy' : 'has completely excised',
 'procedure_has_excised_anatomy' : 'has excised',
 'procedure_has_partially_excised_anatomy' : 'partially excised',
 'procedure_has_target_anatomy' : 'has',
 'process_includes_biological_process' : 'includes',
 'process_initiates_biological_process' : 'initiates',
 'process_involves_gene' : 'involves',
 'product_component_of' : 'Nan',
 'special_category_includes_neoplasm' : 'includes',
 'subset_includes_concept' : 'includes',
 'target_anatomy_has_procedure' : 'has',
 'therapeutic_class_of' : 'Nan',
 'tissue_is_expression_site_of_gene_product' : 'is expression site of',
 'tradename_of' : 'is tradename of ', 
    
}




#relations = [x for x in  list(label2sent.values()) if x != 'Nan']

#models_dir = "/home/ubuntu/nlm/nima/Data/Models/boolq_for_rel/"
#model = RobertaModel.from_pretrained(
#            models_dir,
#            "./model.pt",
#            head="classification",
#            gpt2_encoder_json=f"{models_dir}/encoder.json",
#            gpt2_vocab_bpe=f"{models_dir}/vocab.bpe"
#)




def find_start_end_positions(sentence, answer, start_char):
    answer_tokens = model.encode(answer)[1:-1]
    start_token_idx = len(model.encode(sentence[:start_char].strip())) -1 
    end_token_idx = start_token_idx + len(answer_tokens)

    return start_token_idx, end_token_idx










def process_BioRel_fairseq(dir_):
    print(dir_)
    with open(dir_) as file:
        data = json.load(file)
    

    Relations = [d['relation'] for d in data]
    Relations.remove('NA')
    Start_positions = []
    End_positions = []
    Tokens = []
    for i , data_ in enumerate(data):
        if i % 100 == 0:
            print(i)
        relation = data_['relation'].split("-")
        head = data_['head']['word']
        head_start = data_['head']['start']
        tail = data_['tail']['word']
        tail_start = data_['tail']['start']
        sentence = data_['sentence']

        if converted_relation != 'Nan' and head!='' and tail!='':
            head_start_tok_idx, head_end_tok_idx = find_start_end_positions(sentence, head, head_start)
            tail_start_tok_idx, tail_end_tok_idx = find_start_end_positions(sentence, tail, tail_start)
            Tokens.append(
                model.encode(sentence, relation).tolist()
            )
            Start_positions.append( (head_start_tok_idx, tail_start_tok_idx) )
            End_positions.append( (head_end_tok_idx, tail_end_tok_idx))
        elif relation == "NA":

            fake_relation = Relations[int(np.floor(np.random.uniform() * len(Relations))) ] 
            Tokens.append(model.encode(sentence, fake_relation).tolist() )
            Start_positions.append( (0, 0))
            End_positions.append( (0, 0))


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



def process_BioRel_huggingface_2(dir_, save_dir, file_type='train'):
    with open(dir_) as file:
        data = json.load(file)
    

    Input_ids = []
    Start_positions = []
    End_positions = []
    relations = [ " ".join( d['relation'].split("_") ) for d in data]
    #relations.remove('NA')
    # separating no relation part from relation part
    Data_no_relation = []
    Data_relation = []
    print('separating data!!')
    for d in data:
        relation = d['relation']
        #relation_ = label2sent[ relation ]
        if relation != "NA" and d['head']['word'] != '' and d['tail']['word'] != '':
            d['relation'] = " ".join( d['relation'].split("_") )
            Data_relation.append(d)
        elif relation == 'NA':
            k = np.random.choice(len(relations))
            d['relation'] = relations[k]
            d['head']['word'] = ''
            d['head']['start'] = 0
            d['tail']['start'] = 0
            d['tail']['word'] = ''
            Data_no_relation.append(d)

    print('processing no relation data')
    for i in range(0, len(Data_no_relation), 100):
        print(i)
        examples = Data_no_relation[i : i + 100]
        input_ids, start_positions, end_positions = preprocess_function(examples)
        Input_ids.append(input_ids)
        Start_positions.extend(start_positions)
        End_positions.extend(end_positions)

    print('processing relation data')

    for i in range(0, len(Data_relation), 100):
        print(i)
        examples = Data_relation[i : i + 100]
        input_ids, start_positions, end_positions = preprocess_function(examples)
        Input_ids.append(input_ids)
        Start_positions.extend(start_positions)
        End_positions.extend(end_positions)

    Input_ids = torch.cat(Input_ids, axis = 0)
    print(Input_ids.shape, len(Start_positions), len(End_positions))

    print('saving Data')
    torch.save(Input_ids, f"{save_dir}/input_ids_{file_type}.pt")
    np.save(f"{save_dir}/start_positions_{file_type}", Start_positions)
    np.save(f"{save_dir}/end_positions_{file_type}", End_positions)








def process_BioRel_huggingface(dir_, save_dir, file_type='train'):
    with open(dir_) as file:
        data = json.load(file)
    

    Input_ids = []
    Start_positions = []
    End_positions = []

    # separating no relation part from relation part
    Data_no_relation = []
    Data_relation = []

    print('separating data!!')
    for d in data:
        relation = d['relation']

        if relation != "NA" and d['head']['word'] != '' and d['tail']['word'] != '':
            d['relation'] = d['relation'].split()
            Data_relation.append(d)
        elif relation == 'NA':
            k = np.random.choice(len(relations))
            d['relation'] = relations[k]
            d['head']['word'] = ''
            d['head']['start'] = 0
            d['tail']['start'] = 0 
            d['tail']['word'] = ''
            Data_no_relation.append(d)

    print('processing no relation data')
    for i in range(0, len(Data_no_relation), 100):
        print(i)
        examples = Data_no_relation[i : i + 100]
        input_ids, start_positions, end_positions = preprocess_function(examples)
        Input_ids.append(input_ids)
        Start_positions.extend(start_positions)
        End_positions.extend(end_positions)

    print('processing relation data')

    for i in range(0, len(Data_relation), 100):
        print(i)
        examples = Data_relation[i : i + 100]
        input_ids, start_positions, end_positions = preprocess_function(examples)
        Input_ids.append(input_ids)
        Start_positions.extend(start_positions)
        End_positions.extend(end_positions)

    Input_ids = torch.cat(Input_ids, axis = 0)
    print(Input_ids.shape, len(Start_positions), len(End_positions))

    print('saving Data')
    torch.save(Input_ids, f"{save_dir}/input_ids_{file_type}.pt")
    np.save(f"{save_dir}/start_positions_{file_type}", Start_positions)
    np.save(f"{save_dir}/end_positions_{file_type}", End_positions)





def main(dir_, save_dir="", file_type='train'):

    Tokens, Start_positions, End_positions = process_BioRel_huggingface(dir_)
    
    np.save(save_dir + f"/start_positions_{file_type}", Start_positions )
    np.save(save_dir + f"/end_positions_{file_type}", End_positions )
    
    with open(save_dir + f"not_collated_{file_type}_set_tokens", "wb") as f:
        pickle.dump(Tokens, f)


    #short versions
    

if __name__ == "__main__":
    """
    dir_ = '/home/ubuntu/nlm/nima/Data/BioRel/'
    dir_dev = f'{dir_}/dev.json'
    dir_train = f'{dir_}/train.json'
    save_dir = f'{dir_}/IO-pre-processed-huggingface/'
    #process_BioRel_huggingface(dir_dev, save_dir, file_type='dev')
    process_BioRel_huggingface(dir_train, save_dir, file_type='train')
    """
 

    """
    print('processing train data')
    dir_ = '/home/ubuntu/nlm/nima/Data/BioRel/'å
    dir_train = f'{dir_}/train.json'
    save_dir = f'{dir_}/IO-pre-processed/'
    main(dir_train, save_dir, file_type='train')


    dir_dev = f'{dir_}/dev.json'
    main(dir_dev, save_dir, file_type='dev')    
    """


    """
    dir_ = '/home/ubuntu/nlm/nima/Data/BioRel/'
    dir_dev = f'{dir_}/dev.json'
    dir_train = f'{dir_}/train.json'
    save_dir = f'{dir_}/IO-pre-processed-huggingface/type2/'
    process_BioRel_huggingface_2(dir_dev, save_dir, file_type='dev')
    process_BioRel_huggingface_2(dir_train, save_dir, file_type='train')
    """
    
    dir_ = '/home/ubuntu/nlm/nima/Data/BioRel/'
    dir_dev = f'{dir_}/dev_set_short.json'
    dir_train = f'{dir_}/train_set_short.json'
    save_dir = f'{dir_}/IO-pre-processed-huggingface/type3/'
    process_BioRel_huggingface_2(dir_dev, save_dir, file_type='dev')
    process_BioRel_huggingface_2(dir_train, save_dir, file_type='train')
    
    