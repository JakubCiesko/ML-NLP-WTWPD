import torch 

class GAN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def build_gan(**kwargs) -> GAN: 
    return GAN(**kwargs)