import torch


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
        
        smoothing_coef = self.gan.discriminator.smoothing_coeff if self.gan.discriminator.smoothing_coeff else 0
        discriminator_losses, mapping_losses = [], []
        
        for epoch in range(1, num_epochs + 1):
            mapping_loss_val, discriminator_loss_val = 0, 0

            for batch_i, (source_emb, target_emb) in enumerate(dataloader, start=1):
                source_emb, target_emb = source_emb.to(self.device), target_emb.to(self.device)
                bs = source_emb.size(0)
                
                with torch.no_grad():   #create fake target embeddings
                    source_emb = self.gan.mapping(source_emb)
                
                discriminator_input = torch.cat([source_emb, target_emb], 0).to(self.device) # concat source & target emb
                discriminator_labels = torch.FloatTensor(2 * bs).zero_().to(self.device)
                
                # Smoothing
                discriminator_labels[:bs] = 1 - smoothing_coef #first batch_size embeddings come from mapping
                discriminator_labels[bs:] = smoothing_coef #second batch:size embeddings come from the original distribution
                discriminator_labels = discriminator_labels.unsqueeze(1)
                
                # Discriminator Training
                self.gan.discriminator.train()
                # Forward pass (Discriminator)
                preds = self.gan.discriminator(discriminator_input)
                discriminator_loss = self.criterion_discriminator(preds, discriminator_labels)
                # Backward pass (Discriminator)
                self.optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_discriminator.step()
                discriminator_loss_val += discriminator_loss.data.item()
                
                # Mapping Training 
                self.gan.discriminator.eval()
                self.gan.mapping.train()
                # Forward pass
                preds = self.gan.discriminator(discriminator_input)
                # Backward pass
                mapping_loss = self.criterion_mapping(preds, 1 - discriminator_labels)
                self.optimizer_mapping.zero_grad()
                mapping_loss.backward()
                self.optimizer_mapping.step()
                mapping_loss_val += mapping_loss.data.item()
                self.gan.mapping.eval()
                self.gan.mapping.normalize()

            # Record losses
            mapping_losses.append(mapping_loss_val / batch_i)
            discriminator_losses.append(discriminator_loss_val / batch_i)
            
            # Logging
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}:\nDiscriminator loss: {discriminator_losses[-1]}\tMapping loss: {mapping_losses[-1]}")
        return discriminator_losses, mapping_losses
        