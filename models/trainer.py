import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Trainer:
    def __init__(self, gan, dataloader, criterion_mapping, criterion_discriminator, lr_mapping=0.01, lr_discriminator=0.01):
        self.gan = gan
        self.dataloader = dataloader
        self.criterion_mapping = criterion_mapping
        self.criterion_discriminator = criterion_discriminator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan.to(self.device)

        self.optimizer_mapping = optim.SGD(self.gan.mapping.parameters(), lr=lr_mapping)
        self.optimizer_discriminator = optim.SGD(self.gan.discriminator.parameters(), lr=lr_discriminator)

    def train(self, num_epochs, log_interval=10):
        mapping_losses = []
        discriminator_losses = []

        for epoch in range(num_epochs):
            mapping_loss_val = 0
            discriminator_loss_val = 0

            for batch_i, (source_embeddings, target_embeddings) in enumerate(self.dataloader):
                source_embeddings = source_embeddings.to(self.device)
                target_embeddings = target_embeddings.to(self.device)

                # Forward pass for discriminator
                fake_embeddings = self.gan.mapping(source_embeddings)
                discriminator_input = torch.cat((fake_embeddings, target_embeddings), 0)
                discriminator_labels = torch.cat((torch.ones(fake_embeddings.size(0)), torch.zeros(target_embeddings.size(0))), 0).to(self.device)
                preds = self.gan.discriminator(discriminator_input).squeeze()
                discriminator_loss = self.criterion_discriminator(preds, discriminator_labels)

                # Backward pass and optimization for discriminator
                self.optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_discriminator.step()
                discriminator_loss_val += discriminator_loss.item()

                # Forward pass for mapping
                mapped_source = self.gan.mapping(source_embeddings)
                preds = self.gan.discriminator(mapped_source).squeeze()
                mapping_loss = self.criterion_mapping(preds, torch.ones(source_embeddings.size(0)).to(self.device))

                # Backward pass and optimization for mapping
                self.optimizer_mapping.zero_grad()
                mapping_loss.backward()
                self.optimizer_mapping.step()
                mapping_loss_val += mapping_loss.item()

                # Orthogonalization step
                mapping_tensor = self.gan.mapping.W.weight.data
                mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))
                self.gan.mapping.eval()

            mapping_losses.append(mapping_loss_val / (batch_i + 1))
            discriminator_losses.append(discriminator_loss_val / (batch_i + 1))

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}:\nDiscriminator loss: {discriminator_losses[-1]}\tMapping loss: {mapping_losses[-1]}")

        return discriminator_losses, mapping_losses