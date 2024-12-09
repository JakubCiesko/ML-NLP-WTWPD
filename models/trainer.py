import torch
import torch.nn as nn 
import torch.nn.functional as F



class Trainer():
    
    def __init__(self, gan, optimizer_mapping, optimizer_discriminator, criterion_mapping, criterion_discriminator, scheduler_mapping=None, scheduler_discriminator=None, device='cpu'):
        self.device = device
        self.gan = gan.to(device)
        self.optimizer_mapping = optimizer_mapping
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion_mapping = criterion_mapping
        self.criterion_discriminator = criterion_discriminator
        self.scheduler_mapping = scheduler_mapping
        self.scheduler_discriminator = scheduler_discriminator

    def train(self, dataloader, num_epochs, log_interval=10):
        self.gan.train()
        smoothing_coef = self.gan.discriminator.smoothing_coeff if self.gan.discriminator.smoothing_coeff else 0
        discriminator_losses, mapping_losses = [], []
        for epoch in range(1, num_epochs + 1):
            mapping_loss_val, discriminator_loss_val = 0, 0
            for batch_i, (source_emb, target_emb) in enumerate(dataloader, start=1):
                source_emb, target_emb = source_emb.to(self.device), target_emb.to(self.device)
                bs = source_emb.size(0)
                with torch.no_grad():
                    source_emb = self.gan.mapping(source_emb)
                discriminator_input = torch.cat([source_emb, target_emb], 0).to(self.device) # concat source & target emb
                discriminator_labels = torch.FloatTensor(2 * bs).zero_().to(self.device)
                
                # Smoothing
                
                discriminator_labels[:bs] = 1 - smoothing_coef
                discriminator_labels[bs:] = smoothing_coef
                discriminator_labels = discriminator_labels.unsqueeze(1)
                
                # Discriminator Training
                
                self.gan.discriminator.train()
                preds = self.gan.discriminator(discriminator_input)
                loss = self.criterion_discriminator(preds, discriminator_labels)
                self.optimizer_discriminator.zero_grad()
                loss.backward()
                self.optimizer_discriminator.step()
                discriminator_loss_val += loss.data.item()
                
                # Mapping Training 
                
                self.gan.discriminator.eval()
                self.gan.mapping.train()
                preds = self.gan.discriminator(discriminator_input)
                
                loss = self.criterion_mapping(preds, 1 - discriminator_labels)
                
                self.optimizer_mapping.zero_grad()
                loss.backward()
                self.optimizer_mapping.step()
                mapping_loss_val += loss.data.item()
                mapping_tensor = self.gan.mapping.W.weight.data
                mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))
                self.gan.mapping.eval()

            mapping_losses.append(mapping_loss_val / batch_i)
            discriminator_losses.append(discriminator_loss_val / batch_i)
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}:\nDiscriminator loss: {discriminator_losses[-1]}\tMapping loss: {mapping_losses[-1]}")
        return discriminator_losses, mapping_losses
        