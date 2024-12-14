import torch 
import pickle
import argparse
import numpy as np
from models.gan import GAN
from models.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt



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

def load_embeddings(path):
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = np.array([emb["vector"] for emb in embeddings.values()])
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings
    else:
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a GAN model for embedding mapping")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--epoch_size', type=int, default=10_000, help="Number of iterations per epoch")
    parser.add_argument('--lr_mapping', type=float, default=0.1, help="Learning rate for the mapping optimizer")
    parser.add_argument('--lr_discriminator', type=float, default=0.1, help="Learning rate for the discriminator optimizer")
    parser.add_argument('--log_interval', type=int, default=10, help="Interval for logging training progress")
    parser.add_argument('--mapping_decay', type=float, default=1., help="LR decay for mapping")
    parser.add_argument('--discriminator_steps', type=int, default=1, help="Multiple of training steps for discriminator")
    parser.add_argument('--discriminator_decay', type=float, default=1., help="LR decay for discriminator")
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=300, help="Dimensionality of the embeddings")
    parser.add_argument('--hidden_dim', type=int, default=2048, help="Hidden dimension for the discriminator")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for the discriminator")
    parser.add_argument('--smoothing_coeff', type=float, default=0.2, help="Smoothing coefficient for the discriminator")
    parser.add_argument('--leaky_relu_slope', type=float, default=0.1, help="Slope of LeakyReLU activation")
    parser.add_argument('--weight_init', type=str, choices=[
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 
            'orthogonal', 'uniform', 'normal', 'zeros', 'ones'
        ], 
        #default='xavier_uniform',
        help="Weight initialization method for the network (choices: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'uniform', 'normal', 'zeros', 'ones')"
    )
    parser.add_argument('--optimizer_mapping', type=str, choices=['sgd', 'adam'], default='sgd', 
                    help="Optimizer for the mapping network (choices: 'sgd', 'adam')")
    parser.add_argument('--optimizer_discriminator', type=str, choices=['sgd', 'adam'], default='sgd', 
                    help="Optimizer for the discriminator network (choices: 'sgd', 'adam')")



    # Input data
    parser.add_argument('--src_emb_file', type=str, default=None, help="Path to the source embeddings file")
    parser.add_argument('--tgt_emb_file', type=str, default=None, help="Path to the target embeddings file")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of synthetic samples (if no files provided)")

    # Device argument (CPU or CUDA)
    parser.add_argument('--device', type=str, default="cpu", help="Device to run the model on ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Load embeddings
    if args.src_emb_file and args.tgt_emb_file:
        print(f"Loading embeddings from {args.src_emb_file} and {args.tgt_emb_file}...")
        # Replace this with the actual file loading logic
        source_embeddings = load_embeddings(args.src_emb_file)[:args.num_samples]
        target_embeddings = load_embeddings(args.tgt_emb_file)[:args.num_samples]
    else:
        print("Generating synthetic embeddings...")
        source_embeddings, target_embeddings = generate_synthetic_data(
            num_samples=args.num_samples, embedding_dim=args.embedding_dim
        )
    
    # Create DataLoader
    #dataset = TensorDataset(source_embeddings, target_embeddings)
    #train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # Initialize GAN model
    gan = GAN(
        input_dim=args.embedding_dim,
        output_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        smoothing_coeff=args.smoothing_coeff,
        leaky_relu_slope=args.leaky_relu_slope,
        initialization=args.weight_init,
        bias=None
    )

    # Optimizers
    if args.optimizer_mapping == 'adam':
        mapping_optimizer = torch.optim.Adam(gan.mapping.parameters(), lr=args.lr_mapping)
    else:
        mapping_optimizer = torch.optim.SGD(gan.mapping.parameters(), lr=args.lr_mapping)

    if args.optimizer_discriminator == 'adam':
        discriminator_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=args.lr_discriminator)
    else:
        discriminator_optimizer = torch.optim.SGD(gan.discriminator.parameters(), lr=args.lr_discriminator)

    # Schedulers
    scheduler_mapping = torch.optim.lr_scheduler.ExponentialLR(mapping_optimizer, gamma=args.mapping_decay)
    scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=args.discriminator_decay)

    # Loss functions
    criterion_mapping = torch.nn.BCELoss()
    criterion_discriminator = torch.nn.BCELoss()

    # Trainer
    trainer = Trainer(
        gan=gan,
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        optimizer_mapping=mapping_optimizer,
        optimizer_discriminator=discriminator_optimizer,
        criterion_mapping=criterion_mapping,
        criterion_discriminator=criterion_discriminator,
        scheduler_mapping=scheduler_mapping,
        scheduler_discriminator=scheduler_discriminator, 
        device=args.device
    )
    
    # Train 
    d_l, m_l = trainer.train(
        num_epochs=args.num_epochs,
        iterations_per_epoch=args.epoch_size,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        discriminator_steps=args.discriminator_steps
    )

    # Plot losses
    plt.plot(d_l, label="Discriminator Loss")
    plt.plot(m_l, label="Mapping Loss")
    plt.title("Model loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
