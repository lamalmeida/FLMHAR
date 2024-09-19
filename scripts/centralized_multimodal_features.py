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
from models.fusion_layer import FusionLayer
from experiments.train_encoders import train_contrastive
from experiments.train_classification import train_classification
from experiments.train_decision_fusion import train_decision_fusion
from experiments.train_features_fusion import train_features_fusion 
from utils.utils import get_args
from experiments.evaluation import evaluate_multimodal_features

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

def load_datasets(filename='preprocessed_data.pkl'):
    with open(filename, 'rb') as f:
        train_dataset, val_dataset, test_dataset = pickle.load(f)
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.num_classes = num_classes[args.protocol]

    all_mods = args.modalities.split('|')
    encoders = {}
    optimizers = {}
    classification_heads = {}
    params = []
    for mod in all_mods:
        encoders[mod] = make_encoders(
                mod,
                input_shapes[mod],
                args,
                # pre_trained_path='ckpt/encoders/intermediate/lidar_epoch_33_valloss_0.19.pth'
            ).to(args.device)
        # summary(encoders[mod], input_size=input_shapes[mod])

        optimizers[mod] = torch.optim.Adam(encoders[mod].parameters(), lr=args.contrastive_learning_rate)

    
    classification_final = ClassificationHead(args.d_model * len(all_mods), args.num_classes).to(args.device)
    #classification_final = FusionLayer(args.num_classes * len(all_mods), args.num_classes).to(args.device)
    params.extend(list(classification_final.parameters()))

    train_loader = {}
    val_loader = {}
    test_loader = {}
    for mod in all_mods:
        start_time = time.time()
        args.modalities = mod
        train_dataset, val_dataset, test_dataset = load_datasets(filename=mod + '.pkl')
    
        train_loader[mod] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader[mod] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader[mod] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f'Created {mod} data loaders in {time.time() - start_time:.2f}s')

        # train_contrastive(encoders[mod], optimizers[mod], train_loader[mod], val_loader[mod], epochs=args.contrastive_epochs, args=args)
    encoders['mmwave'].load_state_dict(torch.load('ckpt/encoders/intermediate/mmwave_epoch_97_valloss_0.04.pth'))
    encoders['wifi-csi'].load_state_dict(torch.load('ckpt/encoders/intermediate/wifi-csi_epoch_97_valloss_0.90.pth'))

    optimizer = torch.optim.Adam(params, lr=args.classification_learning_rate)

    train_features_fusion(encoders, classification_final, optimizer, train_loader, val_loader, args)

    evaluate_multimodal_features(encoders, classification_final, test_loader)
    





