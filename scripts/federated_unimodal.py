import torch
import torch.nn as nn
import time
import numpy as np

from collections import defaultdict
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

from mmfi_lib.mmfi import make_datasets, make_dataloader
from models.encoder import make_encoders
from models.classification import ClassificationHead, CombinedModel
from experiments.train_encoders import train_contrastive
from experiments.train_classification import train_classification
from utils.utils import get_args

input_shapes = {
    'lidar': (297, 1660, 3),        # lidar shape
    'infra1': (297, 17, 2),         # infra1 shape
    'infra2': (297, 17, 2),         # infra2 shape
    'mmwave': (297, 112, 5),        # mmWave shape
    'rgb': (297, 17, 2),            # RGB shape
    'wifi-csi': (297, 1140, 3),     # wifi shape
    'depth': (297, 640, 480)        # depth shape
}
num_classes = {
    'protocol1': 14,
    'protocol2': 13,
    'protocol3': 27,
}  

class SubjectClient:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def fit(self, epochs=5):
        self.model = train(self.model, self.dataloader, epochs)
        return self.model
    
def split_by_environment(dataloader):
    environment_data = defaultdict(list)
    for batch in dataloader:
        scenes = batch['scene']
        for i, scene in enumerate(scenes):
            data_point = {key: value[i] for key, value in batch.items() if key not in ['idx', 'modality']}
            environment_data[scene].append(data_point)

    environment_dataloaders = {}
    for env, data in environment_data.items():
        data_dict = defaultdict(list)
        for data_point in data:
            for key, value in data_point.items():
                data_dict[key].append(value)

        input_list = []
        output_list = []
        for key in data_dict:
            if 'input_' in key:
                input_list.append(torch.stack(data_dict[key]).to(device=torch.device('mps'), dtype=torch.float32))
            elif key == 'action':
                output_list.append(torch.stack(data_dict[key]).to(device=torch.device('mps'), dtype=torch.float32))
        
        if len(input_list) > 1:
            input = torch.cat(input_list, dim=1)
        else:
            input = input_list[0]
        
        output = output_list[0]
        
        tensor_dataset = TensorDataset(input, output)
        
        numerical_dataloader = DataLoader(tensor_dataset, batch_size=128, shuffle=True)
        
        environment_dataloaders[env] = numerical_dataloader

    return environment_dataloaders

def create_environment_dataloaders(args):
    train_dataset, val_dataset, test_dataset = make_datasets(args)

    rng_generator = torch.manual_seed(0)
    train_loader = make_dataloader(train_dataset, True, rng_generator, args.batch_size)
    val_loader = make_dataloader(val_dataset, False, rng_generator, args.batch_size)
    test_loader = make_dataloader(test_dataset, False, rng_generator, args.batch_size)

    train_loader = split_by_environment(train_loader)
    # val_loader = split_by_environment(val_loader)
    # test_loader = split_by_environment(test_loader)

    return train_loader, val_loader, test_loader

def federated_averaging(clients, global_model, rounds):
    for round_num in range(rounds):
        global_state_dict = global_model.state_dict()

        # Send global weights to each client
        for client in clients:
            client.model.load_state_dict(global_state_dict)

        local_models = []

        for client in clients:
            local_model = client.fit(epochs=1)
            local_models.append(local_model.state_dict())

        # Average the parameters
        for key in global_state_dict:
            tensors = [local_model[key].to(dtype=torch.float32) for local_model in local_models]
            global_state_dict[key] = torch.stack(tensors).mean(0)
        
        global_model.load_state_dict(global_state_dict)
        print(f"Round {round_num+1} complete")


def train(model, dataloader, epochs=3, lr=0.001):
    model.classification_head.train()
    model.encoder.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        total_loss = []
        for batch in dataloader:
            inputs = batch[0]
            target = batch[1]

            optimizer.zero_grad()

            output = model(inputs)

            loss = criterion(output, target)
            total_loss.append(loss.item())

            loss.backward()
            optimizer.step()

    return model

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.num_classes = num_classes[args.protocol]

    all_mods = args.modalities.split('|')
    for mod in all_mods:
        start_time = time.time()
        args.modalities = mod
        train_loader, val_loader, test_loader = create_environment_dataloaders(args)
        print(f'Created {mod} data loaders in {time.time() - start_time:.2f}s')

        edge_models = {}
        for edge in train_loader.keys():
            encoder = make_encoders(
                mod,
                input_shapes[mod],
                args,
                pre_trained_path='ckpt/encoders/mmwave_epoch_40_valloss_2.32.pth'
            ).to(args.device)

            classification_head = ClassificationHead(args.d_model, args.num_classes).to(args.device)

            edge_models[edge] = CombinedModel(encoder, classification_head).to(args.device)

        clients = [SubjectClient(edge_models[scene], dataloader) for (scene, dataloader) in train_loader.items()]

        encoder = make_encoders(
            mod,
            input_shapes[mod],
            args,
            pre_trained_path='ckpt/encoders/mmwave_epoch_40_valloss_2.32.pth'
        ).to(args.device)

        classification_head = ClassificationHead(args.d_model, args.num_classes).to(args.device)

        global_model = CombinedModel(encoder, classification_head).to(args.device)
        
        for i in range(100):
            print(i)
            federated_averaging(clients, global_model, rounds=5)

            metrics = MetricCollection([
                        Accuracy(num_classes=args.num_classes, average='weighted', task='multiclass'),
                        Precision(num_classes=args.num_classes, average='weighted', task='multiclass'),
                        Recall(num_classes=args.num_classes, average='weighted', task='multiclass'),
                        F1Score(num_classes=args.num_classes, average='weighted', task='multiclass')
                    ]).to(args.device)
            for data in val_loader:
                global_model.eval()
                inferance_data = data['input_' + args.modalities].to(args.device)
                labels = torch.tensor(data['action'], dtype=torch.long).to(args.device)
                outputs = global_model(inferance_data)
                metrics.update(outputs, labels)
            metric = metrics.compute()
            print(f'Accuracy: {metric["MulticlassAccuracy"].item():.4f}, F1: {metric["MulticlassF1Score"].item():.4f}, Precision: {metric["MulticlassPrecision"].item():.4f}, Recall: {metric["MulticlassRecall"].item():.4f}')

        metrics = MetricCollection([
                    Accuracy(num_classes=args.num_classes, average='weighted', task='multiclass'),
                    Precision(num_classes=args.num_classes, average='weighted', task='multiclass'),
                    Recall(num_classes=args.num_classes, average='weighted', task='multiclass'),
                    F1Score(num_classes=args.num_classes, average='weighted', task='multiclass')
                ]).to(args.device)
        for data in test_loader:
            global_model.eval()
            inferance_data = data['input_' + args.modalities].to(args.device)
            labels = torch.tensor(data['action'], dtype=torch.long).to(args.device)
            outputs = global_model(inferance_data)
            metrics.update(outputs, labels)
        metric = metrics.compute()
        print(f'Accuracy: {metric["MulticlassAccuracy"].item():.4f}, F1: {metric["MulticlassF1Score"].item():.4f}, Precision: {metric["MulticlassPrecision"].item():.4f}, Recall: {metric["MulticlassRecall"].item():.4f}')



