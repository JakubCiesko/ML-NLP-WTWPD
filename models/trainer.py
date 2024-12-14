import os
import scipy
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

    def train(self, num_epochs, iterations_per_epoch, batch_size, discriminator_steps, mapping_steps, n_refinement, save_after_n_epoch, checkpoint_dir, log_interval=10):
        if save_after_n_epoch and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        discriminator_losses, mapping_losses = [], []
        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Progress")
        #iteration_bar = tqdm(range(iterations_per_epoch), leave=False, desc="Iteration")
        for epoch in epoch_bar:
            mapping_loss_val, discriminator_loss_val = 0, 0
            for _ in range(iterations_per_epoch):
                for _ in range(discriminator_steps):
                    # Discriminator training
                    self.gan.discriminator.train()
                    self.optimizer_discriminator.zero_grad()
                    discriminator_input, discriminator_labels = self.get_xy(batch_size)
                    discriminator_loss_val += self.discriminator_step(discriminator_input, discriminator_labels)

                # Mapping Training 
                for _ in range(mapping_steps):
                    self.gan.discriminator.eval()
                    self.optimizer_mapping.zero_grad()
                    discriminator_input, discriminator_labels = self.get_xy(batch_size)
                    mapping_labels = 1 - discriminator_labels     
                    mapping_loss_val += self.mapping_step(discriminator_input, mapping_labels, normalize=True)
            # Manipulate LR
            if self.scheduler_discriminator:
                self.scheduler_discriminator.step()
            if self.scheduler_mapping:
                self.scheduler_mapping.step()
            # Record losses
            mapping_losses.append(mapping_loss_val / (iterations_per_epoch*mapping_steps))
            discriminator_losses.append(discriminator_loss_val / (iterations_per_epoch*discriminator_steps))
            
            # Logging
            if epoch % log_interval == 0:
                epoch_bar.set_description(
                    f"Epoch {epoch}: D_loss={discriminator_losses[-1]:.4f}, M_loss={mapping_losses[-1]:.4f}, "
                    f"D_lr={self.optimizer_discriminator.param_groups[0]['lr']:.6f}, "
                    f"M_lr={self.optimizer_mapping.param_groups[0]['lr']:.6f}"
                )
            if save_after_n_epoch and checkpoint_dir:
                if epoch % save_after_n_epoch == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                    self.gan.save_checkpoint(epoch, discriminator_losses, mapping_losses, checkpoint_path)
        for _ in range(n_refinement):
            self.procrustes(self.source_embeddings, self.target_embeddings)
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

    def procrustes(self, source_emb, target_emb):
        W = self.gan.mapping.W.weight
        M = target_emb.transpose(0, 1).mm(source_emb).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.data.copy_(torch.from_numpy(U.dot(V_t)).type_as(W).detach())