import torch
import math

class Mapping(torch.nn.Module):
    def __init__(self, source_embedding_size, target_embedding_size, bias=False, initialization=None):
        super().__init__()
        self.W = torch.nn.Linear(source_embedding_size, target_embedding_size, bias=bias)
        self.use_bias = bias
        if initialization:
            print(f"Initializing weights: {initialization}")
            self.weight_init(initialization)
       
    def forward(self, source_embeddings):
        return self.W(source_embeddings)
    
    def weight_init(self, initialization):
        init_methods = {
            "xavier_uniform": torch.nn.init.xavier_uniform_,
            "xavier_normal": torch.nn.init.xavier_normal_,
            "kaiming_uniform": torch.nn.init.kaiming_uniform_,
            "kaiming_normal": torch.nn.init.kaiming_normal_,
            "orthogonal": torch.nn.init.orthogonal_,
            "uniform": torch.nn.init.uniform_,
            "normal": torch.nn.init.normal_,
            "zeros": torch.nn.init.zeros_,
            "ones": torch.nn.init.ones_,
        }
        init_fn = init_methods.get(initialization, lambda x: torch.nn.init.kaiming_uniform_(x, a=math.sqrt(5)))
        init_fn(self.W.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.W.bias)

    def normalize(self):
        mapping_tensor = self.W.weight.data
        mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))