import torch

class QAModel(torch.nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.io_head = torch.nn.Linear(1024, 2)


    def forward(self, batch):
        embeddings = self.encoder(**batch).last_hidden_state
        output = self.io_head(embeddings)
        

        return output

    def criterion(self, Batch, label):
        
        loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
        logits = self.forward(Batch).squeeze()
        loss = loss_func(
            logits.reshape(-1, 2),
            labels.reshape(-1, 1).squeeze().long()
        )
        logging_output = {
            "status": "something"
        }