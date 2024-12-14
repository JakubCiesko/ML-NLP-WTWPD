import torch
from .discriminator import Discriminator
from .mapping import Mapping

class GAN(torch.nn.Module):
    def __init__(self, num_hidden_layers=2, input_dim=300, output_dim=300, hidden_dim=2048, dropout_rate=0.1, smoothing_coeff=0.2, leaky_relu_slope=0.1, bias=None, initialization=None):
        super().__init__()
        self.mapping = Mapping(input_dim, output_dim, bias=bias, initialization=initialization)
        self.discriminator = Discriminator(num_hidden_layers, input_dim, hidden_dim, dropout_rate, smoothing_coeff, leaky_relu_slope)

    def forward(self, x, y): 
        mapped_source = self.mapping(x)
        discriminator_source = self.discriminator(mapped_source)
        discriminator_target = self.discriminator(y)
        return {
            'mapped_source': mapped_source,
            'discriminator_source': discriminator_source,
            'discriminator_target': discriminator_target
        }

    def save_checkpoint(self, epoch, discriminator_losses, mapping_losses, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'discriminator_losses': discriminator_losses,
            'mapping_losses': mapping_losses
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
