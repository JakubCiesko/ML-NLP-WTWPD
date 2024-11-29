import torch
import torch.nn as nn 
import torch.nn.functional as F



class Trainer():
    
    def __init__(self, gan, optimizer_mapping, optimizer_discriminator, criterion, scheduler_mapping=None, scheduler_discriminator=None, device='cpu'):
        self.gan = gan.to(device)
        self.optimizer_mapping = optimizer_mapping
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion = criterion
        self.scheduler_mapping = scheduler_mapping
        self.scheduler_discriminator = scheduler_discriminator
        self.device = device

    def train(self, dataloader, num_epochs, log_interval=10):
        self.gan.train()
        smoothing_coef = self.gan.discriminator.smoothing_coeff if self.gan.discriminator.smoothing_coeff else 0
        discriminator_losses, mapping_losses = [], []
        for epoch in range(1, num_epochs + 1):
            for batch_i, (source_emb, target_emb) in enumerate(dataloader, start=1):
                bs = source_emb.size(0)
                with torch.no_grad():
                    source_emb = self.gan.mapping(source_emb)
                source_emb, target_emb = source_emb.to(self.device), target_emb.to(self.device)
                discriminator_input = torch.cat([source_emb, target_emb], 0).to(self.device) # concat source & target emb
                discriminator_labels = torch.FloatTensor(2 * bs).zero_()
                
                # Smoothing
                
                discriminator_labels[:bs] = 1 - smoothing_coef
                discriminator_labels[bs:] = smoothing_coef
                discriminator_labels = discriminator_labels.unsqueeze(1)
                
                # Discriminator Training

                self.gan.discriminator.train()
                preds = self.gan.discriminator(discriminator_input)
                loss = F.binary_cross_entropy(preds, discriminator_labels)
                self.optimizer_discriminator.zero_grad()
                loss.backward()
                self.optimizer_discriminator.step()
                discriminator_loss_val = loss.data.item()
                discriminator_losses.append(discriminator_loss_val)
                
                # Mapping Training 
                
                self.gan.discriminator.eval()
                preds = self.gan.discriminator(discriminator_input)
                
                loss = F.binary_cross_entropy(preds, 1 - discriminator_labels)
                #loss = 0.5 * loss
                self.optimizer_mapping.zero_grad()
                loss.backward()
                self.optimizer_mapping.step()
                mapping_loss_val = loss.data.item()
                mapping_losses.append(mapping_loss_val)
               

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}:\nDiscriminator loss: {discriminator_loss_val}\tMapping loss: {mapping_loss_val}")
        return discriminator_losses, mapping_losses
        