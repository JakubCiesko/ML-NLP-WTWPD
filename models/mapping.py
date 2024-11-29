import torch

class Mapping(torch.nn.Module):
    def __init__(self, source_embedding_size, target_embedding_size):
        super().__init__()
        self.W = torch.nn.Linear(source_embedding_size, target_embedding_size, bias=False)
       
    def forward(self, source_embeddings):
        return self.W(source_embeddings)
    
