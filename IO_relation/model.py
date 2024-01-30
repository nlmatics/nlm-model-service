import torch

class RelModel(torch.nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head_io_head = torch.nn.Linear(1024, 2)
        self.tail_io_head = torch.nn.Linear(1024, 2)
        self.tanh = torch.nn.Tanh()

    def forward(self, batch):
        embeddings = self.encoder(**batch).last_hidden_state
        head_ = self.tanh( self.head_io_head(embeddings) )
        tail_ = self.tanh( self.tail_io_head(embeddings) )
        output = torch.cat( (head_[:, :, None, :], tail_[:, :, None, :]), axis = -2)
        # head_shape : (batch_size x seq_len x2)
        return output #head_, tail_

    def criterion(self, batch, labels_head, labels_tail):


        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        logits_head, logits_tail = self.forward(batch)
        #print(labels_head.shape, labels_tail.shape)
        #print(batch.shape, logits_head.shape, logits_tail.shape)
        labels_head_ = labels_head.reshape(-1, 1).squeeze()
        labels_tail_ = labels_tail.reshape(-1, 1).squeeze()
        logits_head_ = logits_head.reshape(-1, 2)
        logits_tail_ = logits_tail.reshape(-1, 2)
        #print(labels_head.shape, labels_head_.shape)

        loss = ( loss_func(logits_head_, labels_head_.long()) + loss_func(logits_tail_, labels_tail_.long()) ) / 2
        
        accuracy_head = (logits_head.argmax(axis = -1) == labels_head).prod(axis = -1).sum().item()
        accuracy_tail = (logits_tail.argmax(axis = -1) == labels_tail).prod(axis = -1).sum().item()
        
        logging_output= {
            "loss": loss.item(),
            "accuracy_head" : accuracy_head, 
            "accuracy_tail" : accuracy_tail, 
            "accuracy": ( accuracy_head + accuracy_tail ) / 2
        }


        return loss, logging_output