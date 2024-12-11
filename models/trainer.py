import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Trainer():
    def __init__(self, gan, dataloader, optimizer_mapping, optimizer_discriminator, criterion_mapping, criterion_discriminator, scheduler_mapping=None, scheduler_discriminator=None, device='cpu'):
        self.device = device
        self.dataloader = dataloader
        self.gan = gan.to(device)
        self.optimizer_mapping = optimizer_mapping
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion_mapping = criterion_mapping
        self.criterion_discriminator = criterion_discriminator
        self.scheduler_mapping = scheduler_mapping
        self.scheduler_discriminator = scheduler_discriminator

    def train(self, num_epochs, log_interval=10):
        dataloader = self.dataloader
        discriminator_losses, mapping_losses = [], []
        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Progress")
        for epoch in epoch_bar:
            mapping_loss_val, discriminator_loss_val = 0, 0
            for batch_i, (source_emb, target_emb) in enumerate(dataloader, start=1):
                bs = source_emb.size(0)
                source_emb, target_emb = source_emb.to(self.device), target_emb.to(self.device)
                fake_emb = self.gan.mapping(source_emb).to(self.device)
                
                discriminator_input = torch.cat([fake_emb, target_emb], 0).to(self.device) # concat mapped source & target emb
                discriminator_labels = torch.cat((torch.ones(bs), torch.zeros(bs)), 0).to(self.device) # [1,..., 1, 0,....,0]
                
                # Smoothing
                discriminator_labels = self.smooth_labels(discriminator_labels, bs).to(self.device)
                
                # Discriminator Training
                discriminator_loss_val += self.discriminator_step(discriminator_input, discriminator_labels)
                
                # Mapping Training 
                fake_emb = self.gan.mapping(source_emb).to(self.device)
                mapping_input = torch.cat([fake_emb, target_emb], 0).to(self.device)
                mapping_labels = 1 - torch.cat((torch.ones(bs), torch.zeros(bs)), 0).to(self.device)
                mapping_labels = self.smooth_labels(mapping_labels, bs).to(self.device)
                mapping_loss_val += self.mapping_step(mapping_input, mapping_labels, True)
            self.scheduler_discriminator.step()
            self.scheduler_mapping.step()   
            # Record losses
            mapping_losses.append(mapping_loss_val / batch_i)
            discriminator_losses.append(discriminator_loss_val / batch_i)
            
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
        labels[:point] = 1 - smoothing_coef #first batch_size embeddings come from mapping
        labels[point:] = smoothing_coef #second batch:size embeddings come from the original distribution
        labels = labels.unsqueeze(1)
        return labels

    def discriminator_step(self, discriminator_input, discriminator_labels):
        preds = self.gan.discriminator(discriminator_input)
        discriminator_loss = self.criterion_discriminator(preds, discriminator_labels)
        self.optimizer_discriminator.zero_grad()
        discriminator_loss.backward()
        self.optimizer_discriminator.step()
        return discriminator_loss.data.item()

    def mapping_step(self, mapping_input, mapping_labels, normalize=True):
        preds = self.gan.discriminator(mapping_input)
        mapping_loss = self.criterion_mapping(preds, mapping_labels)
        self.optimizer_mapping.zero_grad()
        mapping_loss.backward()
        self.optimizer_mapping.step()
        if normalize:
            self.gan.mapping.normalize()
        return mapping_loss.data.item()

    def compute_average_similarity(self, embeddings, k=10):
        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, -np.inf)  # Ignore self-similarity
        avg_similarities = np.mean(np.sort(similarities, axis=1)[:, -k:], axis=1)
        return avg_similarities

    def compute_csls_score(self, source_embeddings, target_embeddings, k=10):
        source_avg_sim = self.compute_average_similarity(source_embeddings, k)
        target_avg_sim = self.compute_average_similarity(target_embeddings, k)
        
        csls_scores = np.zeros((source_embeddings.shape[0], target_embeddings.shape[0]))
        for i, source_vec in enumerate(source_embeddings):
            for j, target_vec in enumerate(target_embeddings):
                csls_scores[i, j] = 2 * cosine_similarity([source_vec], [target_vec])[0][0] - source_avg_sim[i] - target_avg_sim[j]
        
        return csls_scores

    def find_nearest_neighbors(self, source_embeddings, target_embeddings, k=10):
        csls_scores = self.compute_csls_score(source_embeddings, target_embeddings, k)
        nearest_neighbors = np.argmax(csls_scores, axis=1)
        return nearest_neighbors
