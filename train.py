from models.trainer import Trainer
from models.gan import GAN
from torch.utils.data import DataLoader, TensorDataset
import torch 


def generate_synthetic_data(num_samples=1000, embedding_dim=300):
    """
    Generates synthetic data for source and target embeddings.
    Source embeddings are randomly initialized, and target embeddings
    are transformations of the source embeddings using a fixed linear transformation.
    """
    torch.manual_seed(42)  # For reproducibility
    source_embeddings = torch.randn(num_samples, embedding_dim)
    true_w = torch.randn(embedding_dim, embedding_dim)
    target_embeddings = torch.matmul(source_embeddings, true_w)
    return source_embeddings, target_embeddings


batch_size = 32
num_epochs = 10

source_embeddings, target_embeddings = generate_synthetic_data(embedding_dim=20)
dataset = TensorDataset(source_embeddings, target_embeddings)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



gan = GAN(input_dim=20, output_dim=20, hidden_dim=100, dropout_rate=0.1, smoothing_coeff=0.1)

mapping_optimizer = torch.optim.SGD(gan.mapping.parameters(), lr=0.1)
discriminator_optimizer = torch.optim.SGD(gan.discriminator.parameters(), lr=0.01)


criterion = torch.nn.BCELoss()

trainer = Trainer(
    gan=gan,
    optimizer_mapping=mapping_optimizer,
    optimizer_discriminator=discriminator_optimizer,
    criterion=criterion,
    scheduler_mapping=None,
    scheduler_discriminator=None
)



trainer.train(dataloader=train_loader, num_epochs=num_epochs, log_interval=1)

