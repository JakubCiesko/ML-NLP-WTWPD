import torch 

class Discriminator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def build_discriminator(**kwargs) -> Discriminator:
    return Discriminator(kwargs)