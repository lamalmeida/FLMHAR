import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift

from data.data_augmentation import augment_data
import time
import numpy as np
import gc

class EncoderWrapper(nn.Module):
    def __init__(self, encoder, original_dim, encoded_dim):
        super(EncoderWrapper, self).__init__()
        self.encoder = encoder
        self.linear_layer = nn.Linear(original_dim, encoded_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.linear_layer(encoded)
        return output
    
def cluster_loss(z_i, z_j, device, n_clusters, temperature=0.1):
    batch_size = z_i.shape[0]
    LARGE_NUM = 1e9

    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    logits_aa = torch.matmul(z_i, z_i.T) / temperature
    logits_bb = torch.matmul(z_j, z_j.T) / temperature
    logits_ab = torch.matmul(z_i, z_j.T) / temperature
    logits_ba = torch.matmul(z_j, z_i.T) / temperature

    # logits_aa
    cluster = KMeans(n_clusters=n_clusters)
    pre_class = cluster.fit_predict(z_i.detach().cpu().numpy())

    masks_aa = torch.tensor(pre_class).unsqueeze(1) == torch.tensor(pre_class).unsqueeze(0)
    masks_aa = masks_aa.to(z_i.device).float()
    logits_aa = logits_aa - masks_aa * LARGE_NUM
    masks_ab = masks_aa - torch.eye(batch_size, device=z_i.device)
    logits_ab = logits_ab - masks_ab * LARGE_NUM

    # logits_bb
    cluster = KMeans(n_clusters=n_clusters)
    pre_class = cluster.fit_predict(z_j.detach().cpu().numpy())

    masks_bb = torch.tensor(pre_class).unsqueeze(1) == torch.tensor(pre_class).unsqueeze(0)
    masks_bb = masks_bb.to(z_j.device).float()
    logits_bb = logits_bb - masks_bb * LARGE_NUM
    masks_ba = masks_bb - torch.eye(batch_size, device=z_j.device)
    logits_ba = logits_ba - masks_ba * LARGE_NUM

    labels = torch.arange(batch_size, device=z_i.device)

    logits_a = torch.cat([logits_ab, logits_aa], dim=1)
    logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)

    return loss_a + loss_b

def contrastive_loss(z_i, z_j, device, n_clusters, temperature=0.1):
    batch_size = z_i.shape[0]
    LARGE_NUM = 1e9

    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    
    labels = torch.eye(batch_size * 2)[torch.arange(batch_size)].to(device)
    masks = torch.eye(batch_size)[torch.arange(batch_size)].to(device)

    logits_aa = torch.matmul(z_i, z_i.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(z_j, z_j.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(z_i, z_j.T) / temperature
    logits_ba = torch.matmul(z_j, z_i.T) / temperature

    logits_a = torch.cat([logits_ab, logits_aa], dim=1)
    logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss = torch.mean(loss_a + loss_b)

    return loss

def train_step(mod, model, optimizer, data, device, n_clusters):
    model.train()
    optimizer.zero_grad()

    augmented_data_1 = data # augment_data(data, mod)
    augmented_data_2 = augment_data(data, mod)

    proj_1 = model(augmented_data_1)
    proj_2 = model(augmented_data_2)

    loss = contrastive_loss(proj_1, proj_2, device, n_clusters)
    loss.backward()
    optimizer.step()
    return loss.item()

def validation_step(mod, model, data, device, n_clusters):
    model.eval()

    augmented_data_1 = data # augment_data(data, mod)
    augmented_data_2 = augment_data(data, mod)
    
    proj_1 = model(augmented_data_1)
    proj_2 = model(augmented_data_2)

    loss = contrastive_loss(proj_1, proj_2, device, n_clusters)
    return loss.item()

def train_contrastive(encoder, optimizer, train_loader, val_loader, epochs, args):
    min_vall_loss = 10**9
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = []
        for data in train_loader:
            data[0] = data[0].to(args.device)
            total_loss.append(train_step(args.modalities, encoder, optimizer, data[0], args.device, args.num_classes))
            
        print(f'[{args.modalities:<10}] Epoch {epoch + 1:>2}, Loss: {np.mean(total_loss):.4f} ({time.time() - start_time:.4f})')

        start_time = time.time()
        val_loss = []
        for val_data in val_loader:
            val_data[0] = val_data[0].to(args.device)
            val_loss.append(validation_step(args.modalities, encoder, val_data[0], args.device, args.num_classes))
        
        print(f'[{args.modalities:<10}] Epoch {epoch + 1:>2}, Validation Loss: {np.mean(val_loss):.4f} ({time.time() - start_time:.4f})')
        if np.mean(val_loss) < min_vall_loss:
            min_vall_loss = min(min_vall_loss, np.mean(val_loss))
            file = f'ckpt/encoders/intermediate/{args.modalities}_epoch_{epoch + 1}_valloss_{np.mean(val_loss):.2f}.pth'
            torch.save(encoder.state_dict(), file)
        
        scheduler.step()
        
        gc.collect()
        torch.mps.empty_cache()

    encoder.load_state_dict(torch.load(file))

        


