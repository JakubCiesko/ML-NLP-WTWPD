import torch 

class Mapping(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def build_mapping(**kwargs) -> Mapping:
    return Mapping(**kwargs)