import torch
import numpy as np
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from mmfi_lib.mmfi import make_datasets, make_dataloader
from utils.utils import get_args

input_shapes = {
    'lidar': (297, 1000, 3),        # lidar shape
    'infra1': (297, 17, 2),        # infra1 shape
    'infra2': (297, 17, 2),        # infra2 shape
    'mmwave': (297, 112, 5),       # mmWave shape
    'rgb': (297, 17, 2),           # RGB shape
    'wifi-csi': (297, 1140, 3),    # wifi shape
    'depth': (297, 640, 480)       # depth shape
}

num_classes = {
    'protocol1': 14,
    'protocol2': 13,
    'protocol3': 27,
}

def split_by_frames(train, val, test):
    for mod in args.modalities.split('|'):
        modified_data_train = defaultdict(list)
        modified_data_val = defaultdict(list)
        modified_data_test = defaultdict(list)
        for dataloader in [train, test, val]:
            for batch in dataloader:
                batch_size = len(batch['action'])
                for i in range(batch_size):
                    for j in range(0, 5*33, 33):
                        modified_data_train['action'].append(batch['action'][i])
                        modified_data_train['input_' + mod].append(batch['input_' + mod][i][j:j+33])
                    for j in range(5*33, 7*33, 33):
                        modified_data_val['action'].append(batch['action'][i])
                        modified_data_val['input_' + mod].append(batch['input_' + mod][i][j:j+33])
                    for j in range(7*33, 9*33, 33):
                        modified_data_test['action'].append(batch['action'][i])
                        modified_data_test['input_' + mod].append(batch['input_' + mod][i][j:j+33])


        for key in modified_data_train:
            modified_data_train[key] = torch.stack(modified_data_train[key])
            modified_data_val[key] = torch.stack(modified_data_val[key])
            modified_data_test[key] = torch.stack(modified_data_test[key])
        
        mean = modified_data_train['input_' + mod].mean(dim=(0, 1, 2), keepdim=True)
        std = modified_data_train['input_' + mod].std(dim=(0, 1, 2), keepdim=True)

        modified_data_train['input_' + mod] = (modified_data_train['input_' + mod] - mean) / std
        modified_data_val['input_' + mod] = (modified_data_val['input_' + mod] - mean) / std
        modified_data_test['input_' + mod] = (modified_data_test['input_' + mod] - mean) / std

        input_train_list = []
        output_train_list = []
        input_val_list = []
        output_val_list = []
        input_test_list = []
        output_test_list = []
        for key in modified_data_train:
            if 'input_' in key:
                input_train_list.append(modified_data_train[key])
                input_val_list.append(modified_data_val[key])
                input_test_list.append(modified_data_test[key])
            elif key == 'action':
                output_train_list.append(modified_data_train[key])
                output_val_list.append(modified_data_val[key])
                output_test_list.append(modified_data_test[key])

        input_train_tensor = torch.cat(input_train_list, dim=1)
        input_val_tensor = torch.cat(input_val_list, dim=1)
        input_test_tensor = torch.cat(input_test_list, dim=1)

        output_train_tensor = output_train_list[0]
        output_val_tensor = output_val_list[0]
        output_test_tensor = output_test_list[0]

        tensor_train_dataset = TensorDataset(input_train_tensor, output_train_tensor)
        tensor_val_dataset = TensorDataset(input_val_tensor, output_val_tensor)
        tensor_test_dataset = TensorDataset(input_test_tensor, output_test_tensor)
        
        save_datasets(tensor_train_dataset, tensor_val_dataset, tensor_test_dataset, filename=mod + '.pkl')

def create_dataloaders(args):
    train_dataset, val_dataset, test_dataset = make_datasets(args)
    rng_generator = torch.manual_seed(0)
    train_loader = make_dataloader(train_dataset, True, rng_generator, args.batch_size)
    val_loader = make_dataloader(val_dataset, False, rng_generator, args.batch_size)
    test_loader = make_dataloader(test_dataset, False, rng_generator, args.batch_size) 
    split_by_frames(train_loader, val_loader, test_loader)

def save_datasets(train_dataset, val_dataset, test_dataset, filename='preprocessed_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((train_dataset, val_dataset, test_dataset), f)

if __name__ == '__main__':
    args = get_args()
    create_dataloaders(args)