import torch
from tqdm import tqdm


class Trainer():
    def __init__(self, gan, source_embeddings, target_embeddings, optimizer_mapping, optimizer_discriminator, criterion_mapping, criterion_discriminator, scheduler_mapping=None, scheduler_discriminator=None, device='cpu'):
        self.device = device
        self.source_embeddings=source_embeddings
        self.target_embeddings=target_embeddings
        self.gan = gan.to(device)
        self.optimizer_mapping = optimizer_mapping
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion_mapping = criterion_mapping
        self.criterion_discriminator = criterion_discriminator
        self.scheduler_mapping = scheduler_mapping
        self.scheduler_discriminator = scheduler_discriminator

    def train(self, num_epochs, iterations_per_epoch, batch_size, discriminator_steps, log_interval=10):
        discriminator_losses, mapping_losses = [], []
        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Progress")
        for epoch in epoch_bar:
            mapping_loss_val, discriminator_loss_val = 0, 0
            for iteration in range(iterations_per_epoch):
                for _ in range(discriminator_steps):
                    # Discriminator training
                    self.gan.discriminator.train()
                    self.optimizer_discriminator.zero_grad()
                    discriminator_input, discriminator_labels = self.get_xy(batch_size)
                    discriminator_loss_val += self.discriminator_step(discriminator_input, discriminator_labels)

                # Mapping Training 
                self.gan.discriminator.eval()
                self.optimizer_mapping.zero_grad()
                discriminator_input, discriminator_labels = self.get_xy(batch_size)
                mapping_labels = 1 - discriminator_labels     
                mapping_loss_val += self.mapping_step(discriminator_input, mapping_labels, normalize=True)
        
            self.scheduler_discriminator.step()
            self.scheduler_mapping.step()   
            # Record losses
            mapping_losses.append(mapping_loss_val / (iteration+1))
            discriminator_losses.append(discriminator_loss_val / ((iteration+1)*discriminator_steps))
            
            # Logging
            if epoch % log_interval == 0:
                epoch_bar.set_description(
                    f"Epoch {epoch}: D_loss={discriminator_losses[-1]:.4f}, M_loss={mapping_losses[-1]:.4f}, "
                    f"D_lr={self.optimizer_discriminator.param_groups[0]['lr']:.6f}, "
                    f"M_lr={self.optimizer_mapping.param_groups[0]['lr']:.6f}"
                )
        return discriminator_losses, mapping_losses
        
    def smooth_labels(self, labels, point):
        smoothing_coef = self.gan.discriminator.smoothing_coeff if self.gan.discriminator.smoothing_coeff else 0
        labels[:point] = 1 - smoothing_coef 
        labels[point:] = smoothing_coef 
        labels = labels.unsqueeze(1)
        return labels
    
    def get_xy(self, batch_size):
        source_emb_id = torch.Tensor(batch_size).random_(len(self.source_embeddings)).long()
        target_emb_id = torch.Tensor(batch_size).random_(len(self.source_embeddings)).long()
        source_emb = self.source_embeddings[source_emb_id].to(self.device)
        target_emb = self.target_embeddings[target_emb_id].to(self.device)
        mapped_emb = self.gan.mapping(source_emb)
        x = torch.cat([mapped_emb, target_emb], 0).to(self.device)
        y = torch.Tensor(2*batch_size).zero_().float().to(self.device)
        y = self.smooth_labels(y, batch_size)
        return x, y

    def discriminator_step(self, discriminator_input, discriminator_labels):
        preds = self.gan.discriminator(discriminator_input)
        discriminator_loss = self.criterion_discriminator(preds, discriminator_labels).to(self.device)
        discriminator_loss.backward()
        self.optimizer_discriminator.step()
        return discriminator_loss.data.item()

    def mapping_step(self, discriminator_input, mapping_labels, normalize=True):
        preds = self.gan.discriminator(discriminator_input)
        mapping_loss = self.criterion_mapping(preds, mapping_labels).to(self.device)
        mapping_loss.backward()
        self.optimizer_mapping.step()
        if normalize:
            self.gan.mapping.normalize()
        return mapping_loss.data.item()
