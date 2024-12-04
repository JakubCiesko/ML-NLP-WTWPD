import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim



class Trainer:
    def __init__(self, gan, dataloader, criterion_mapping, criterion_discriminator, lr_mapping=0.0001, lr_discriminator=0.0001):
        self.gan = gan
        self.dataloader = dataloader
        self.criterion_mapping = criterion_mapping
        self.criterion_discriminator = criterion_discriminator
        self.optimizer_mapping = optim.Adam(self.gan.mapping.parameters(), lr=lr_mapping)
        self.optimizer_discriminator = optim.Adam(self.gan.discriminator.parameters(), lr=lr_discriminator)

    def train(self, num_epochs, log_interval=10):
        mapping_losses = []
        discriminator_losses = []

        for epoch in range(num_epochs):
            mapping_loss_val = 0
            discriminator_loss_val = 0

            for batch_i, (source_embeddings, target_embeddings) in enumerate(self.dataloader):
                # Forward pass for discriminator
                discriminator_input = torch.cat((source_embeddings, target_embeddings), 0)
                discriminator_labels = torch.cat((torch.ones(source_embeddings.size(0)), torch.zeros(target_embeddings.size(0))), 0)
                preds = self.gan.discriminator(discriminator_input)
                discriminator_loss = self.criterion_discriminator(preds, discriminator_labels)

                # Backward pass and optimization for discriminator
                self.optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_discriminator.step()
                discriminator_loss_val += discriminator_loss.data.item()

                # Forward pass for mapping
                mapped_source = self.gan.mapping(source_embeddings)
                preds = self.gan.discriminator(mapped_source)
                mapping_loss = self.criterion_mapping(preds, torch.ones(source_embeddings.size(0)))

                # Backward pass and optimization for mapping
                self.optimizer_mapping.zero_grad()
                mapping_loss.backward()
                self.optimizer_mapping.step()
                mapping_loss_val += mapping_loss.data.item()

                # Orthogonalization step
                mapping_tensor = self.gan.mapping.W.weight.data
                mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))
                self.gan.mapping.eval()

            mapping_losses.append(mapping_loss_val / (batch_i + 1))
            discriminator_losses.append(discriminator_loss_val / (batch_i + 1))

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}:\nDiscriminator loss: {discriminator_losses[-1]}\tMapping loss: {mapping_losses[-1]}")

        return discriminator_losses, mapping_losses