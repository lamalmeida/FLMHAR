import torch
import time
import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchsummary import summary
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from mmfi_lib.mmfi import make_datasets, make_dataloader
from models.encoder import make_encoders
from models.classification import ClassificationHead
from experiments.train_encoders import train_contrastive
from experiments.train_classification import train_classification
from utils.utils import get_args
from experiments.evaluation import evaluate_unimodal

input_shapes = {
    'lidar': (33, 1660, 3),        # lidar shape
    'infra1': (297, 17, 2),         # infra1 shape
    'infra2': (297, 17, 2),         # infra2 shape
    'mmwave': (33, 112, 5),        # mmWave shape
    'rgb': (33, 17, 2),            # RGB shape
    'wifi-csi': (33, 1140, 3),     # wifi shape
    'depth': (297, 640, 480)        # depth shape
}
num_classes = {
    'protocol1': 14,
    'protocol2': 13,
    'protocol3': 27,
}  

def evaluate_encoder(encoder, dataloader, num_clusters, device):
    encoder.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            embeddings = encoder(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Standardize the embeddings
    scaler = StandardScaler()
    all_embeddings = scaler.fit_transform(all_embeddings)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(all_embeddings)

    # Calculate clustering metrics
    ari = adjusted_rand_score(all_labels, cluster_labels)
    nmi = normalized_mutual_info_score(all_labels, cluster_labels)

    return {
        "ARI": ari,
        "NMI": nmi,
    }

def load_datasets(filename='preprocessed_data.pkl'):
    with open(filename, 'rb') as f:
        train_dataset, val_dataset, test_dataset = pickle.load(f)
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.num_classes = num_classes[args.protocol]

    all_mods = args.modalities.split('|')
    for mod in all_mods:
        start_time = time.time()
        args.modalities = mod
        train_dataset, val_dataset, test_dataset = load_datasets()
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f'Created {mod} data loaders in {time.time() - start_time:.2f}s')

        encoder = make_encoders(
                mod,
                input_shapes[mod],
                args,
                # pre_trained_path='ckpt/encoders/intermediate/lidar_epoch_33_valloss_0.19.pth'
            ).to(args.device)
        # summary(encoders[mod], input_size=input_shapes[mod])

        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.contrastive_learning_rate)

        train_contrastive(encoder, optimizer, train_loader, val_loader, epochs=args.contrastive_epochs, args=args)

        metrics = evaluate_encoder(encoder, test_loader, args.num_classes, device=torch.device('mps'))
        print("Encoder Evaluation Metrics:", metrics)

        classification_head = ClassificationHead(args.d_model, args.num_classes).to(args.device)

        optimizer = torch.optim.Adam(classification_head.parameters(), lr=args.classification_learning_rate)

        train_classification(encoder, classification_head, optimizer, train_loader, val_loader, epochs=args.classification_epochs, args=args)

        evaluate_unimodal(encoder, classification_head, test_loader)



