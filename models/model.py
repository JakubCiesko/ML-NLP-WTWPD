import torch 

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def build_model(**kwargs) -> Model:
    return Model(kwargs)