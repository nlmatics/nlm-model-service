import torch
import numpy as np
import pickle
from fairseq.data.data_utils import collate_tokens



class Dataset2(torch.utils.data.Dataset):
      #for pickle data
      'Characterizes a dataset for PyTorch'
      def __init__(
                  self, 
                  dir_input_ids,
                  dir_labels_start,
                  dir_labels_end, 
            ):
                  'Initialization'
                  with open(dir_input_ids, 'rb') as file:
                        input_ids = pickle.load(file)
            
                  
                  self.input_ids = [torch.tensor(x) for x in input_ids]
                  self.labels_start = np.load(dir_labels_start)
                  self.labels_end = np.load(dir_labels_end)

            
      
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.input_ids)

      def __getitem__(self, idx):
            'Generates one sample of data'
            labels = torch.zeros([2, 512])
            #head
            labels[0][self.labels_start[idx][0] : self.labels_end[idx][0] ] =\
                  torch.ones(self.labels_end[idx][0]-self.labels_start[idx][0])
            #tail
            labels[1][self.labels_start[idx][1] : self.labels_end[idx][1]] =\
                  torch.ones(self.labels_end[idx][1]-self.labels_start[idx][1])
      
            return self.input_ids[idx][:512], labels[0], labels[1]



def collate_fn_2(data, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
      #suitbale for Dataset2      
      input_ids = collate_tokens([d[0] for d in data], pad_idx = 1).to(device)
      _, seq_length = input_ids.shape
      labels_head = torch.stack([ d[1] for d in data])[:, :seq_length].to(device)
      labels_tail = torch.stack([ d[2] for d in data])[:, :seq_length].to(device)
      labels = [l[1] for l in data]         
      
      
      #input_batch = {
      #      'net_input':{
      #      "src_tokens" : input_ids, 
      #      } , 
      #      "target": torch.LongTensor(labels).to(device), 
      #      "ntokens": input_ids.shape[1]
            
      #}

      return input_ids, labels_head, labels_tail








class Dataset1(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(
      self,
      dir_input_ids,
      dir_start_positions, 
      dir_end_postions
      ):
        'Initialization'
        self.input_ids = torch.load(dir_input_ids)
        self.labels_start = np.load(dir_start_positions)
        self.labels_end = np.load(dir_end_postions)
        
     
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_ids)

  def __getitem__(self, idx):
      'Generates one sample of data'
      'Generates one sample of data'
      labels = torch.zeros([2, 512])
      #head            

      labels[0][self.labels_start[idx][0] : self.labels_end[idx][0] ] =\
            torch.ones(self.labels_end[idx][0]-self.labels_start[idx][0])
      #tail
      labels[1][self.labels_start[idx][1] : self.labels_end[idx][1]] =\
            torch.ones(self.labels_end[idx][1]-self.labels_start[idx][1])


      return self.input_ids[idx][:512], labels[0], labels[1]


def collate_fn_1(data, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
      
      input_ids = torch.stack([d[0] for d in data])
      heads = torch.stack([l[1] for l in data])
      tails = torch.stack([l[2] for l in data])
      paddings = input_ids == 1
      
      number_of_paddings = paddings.sum(axis = -1)
      max_length = 512 - min(number_of_paddings)
      
      attention_mask = torch.ones([input_ids.shape[0], input_ids.shape[1]])
      attention_mask[ np.where(paddings == 1)] = 0

      input_ids = input_ids[:, :max_length]
      attention_mask = attention_mask[:, :max_length]

      input_batch = {
            'input_ids': input_ids.to(device), 
            'attention_mask': attention_mask.to(device)
      }

      return input_batch, heads[:, :max_length].to(device), tails[:, :max_length].to(device)