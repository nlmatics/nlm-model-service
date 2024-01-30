import torch


class RetModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.lin_layer = torch.nn.Linear(1024, 1)
        self.encoder = encoder

    def forward(self, batch):
        torch.set_printoptions(profile="full")
        features = self.encoder(**batch)
        logit = self.lin_layer(features["last_hidden_state"][:, 0])
        return logit


# now we want two encoders
# one for q, one for passage

# have two classes, class Q for question encoder and class P for passage
