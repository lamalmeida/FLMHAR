import os
import scipy.io as scio
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

protocol_1_mapping = {
    'A02': 0, 'A03': 1, 'A04': 2, 'A05': 3, 'A13': 4, 'A14': 5,
    'A17': 6, 'A18': 7, 'A19': 8, 'A20': 9, 'A21': 10, 'A22': 11,
    'A23': 12, 'A27': 13
}

protocol_2_mapping = {
    'A01': 0, 'A06': 1, 'A07': 2, 'A08': 3, 'A09': 4, 'A10': 5,
    'A11': 6, 'A12': 7, 'A15': 8, 'A16': 9, 'A24': 10, 'A25': 11,
    'A26': 12
}

protocol_3_mapping = {f'A{str(i).zfill(2)}': i - 1 for i in range(1, 28)}

def decode_config(args):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    train_form = {}
    val_form = {}
    test_form = {}

    # Limitation to actions (protocol)
    if args.protocol == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif args.protocol == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    
    for subject in all_subjects:
        # Randomly shuffle the indices for the subject's data
        idx = np.random.permutation(len(actions))

        # Determine the split points
        val_size = int(np.floor(args.val_ratio/100 * len(actions)))
        test_size = int(np.floor(args.test_ratio/100 * len(actions)))
        train_size = len(actions) - val_size - test_size

        # Split indices into training, validation, and test sets
        idx_train = idx[:train_size]
        idx_val = idx[train_size:train_size + val_size]
        idx_test = idx[train_size + val_size:]

        # Get actions for each split
        actions_train = [actions[i] for i in idx_train]
        actions_val = [actions[i] for i in idx_val]
        actions_test = [actions[i] for i in idx_test]

        # Add to the final datasets
        train_form[subject] = actions_train
        val_form[subject] = actions_val
        test_form[subject] = actions_test

    dataset_config = {'train_dataset': {'modality': args.modalities,
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': args.modalities,
                                      'split': 'validation',
                                      'data_form': val_form
                                      },
                      'test_dataset': {'modality': args.modalities,
                                        'split': 'test',
                                        'data_form': test_form
                                        }
                     }
    return dataset_config

class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene, _, _ in os.walk(self.data_root):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject, _, _ in os.walk(os.path.join(self.data_root, scene)):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action, _, _ in os.walk(os.path.join(self.data_root, scene, subject)):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions:
                        self.actions[action] = {}
                    if scene not in self.actions[action]:
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene]:
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities:
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality]:
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene]:
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject]:
                            self.modalities[modality][scene][subject][action] = data_path

class MMFi_Dataset(Dataset):
    def __init__(self, data_base, mapping, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.mapping = mapping
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                data_dict = {'modality': self.modality,
                            'scene': self.get_scene(subject),
                            'subject': subject,
                            'action': self.mapping[action],
                            }
                for mod in self.modality:
                    data_dict[mod+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod)
                data_info.append(data_dict)
                
        return data_info

    def read_dir(self, dir):
        _, mod = os.path.split(dir)
        data = []
        if mod in ['infra1', 'infra2', 'rgb']:  # rgb, infra1 and infra2 are 2D keypoints
            for arr_file in sorted(glob.glob(os.path.join(dir, "frame*.npy"))):
                arr = np.load(arr_file).astype(np.float32)
                data.append(arr)
            data = np.array(data)
        elif mod == 'depth':
            for img in sorted(glob.glob(os.path.join(dir, "frame*.png"))):
                _cv_img = (cv2.imread(img, cv2.IMREAD_UNCHANGED) * 0.001).astype(np.float32) # Default depth value is 16-bit
                data.append(_cv_img.T)
            data = np.array(data)
        elif mod == 'lidar':
            max_shape = (1660, 3)
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 3)
                pad_width = ((0, max_shape[0] - data_tmp.shape[0]), (0, 0))
                data_tmp = np.pad(data_tmp, pad_width, mode='constant', constant_values=0).astype(np.float32)
                data.append(data_tmp)
            data = np.array(data)
        elif mod == 'mmwave':
            max_shape = (112, 5)
            for bin_file in sorted(glob.glob(os.path.join(dir, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    raw_data = f.read()
                    data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                    data_tmp = data_tmp.reshape(-1, 5)
                pad_width = ((0, max_shape[0] - data_tmp.shape[0]), (0, 0))
                data_tmp = np.pad(data_tmp, pad_width, mode='constant', constant_values=0).astype(np.float32)
                data.append(data_tmp)
            data = np.array(data)
        elif mod == 'wifi-csi':
            for csi_mat in sorted(glob.glob(os.path.join(dir, "frame*.mat"))):
                data_mat = scio.loadmat(csi_mat)['CSIamp'].astype(np.float32)
                data_mat[np.isinf(data_mat)] = np.nan
                for i in range(10):  
                    temp_col = data_mat[:, :, i]
                    nan_num = np.count_nonzero(temp_col != temp_col)
                    if nan_num != 0:
                        temp_not_nan_col = temp_col[temp_col == temp_col]
                        temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
                data_mat = (data_mat - np.min(data_mat)) / (np.max(data_mat) - np.min(data_mat))
                data_frame = np.array(data_mat)
                data_frame = data_frame.reshape(3, -1).T
                data.append(data_frame)
            data = np.array(data).astype(np.float32)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        sample = {'modality': item['modality'],
                 'scene': item['scene'],
                 'subject': item['subject'],
                 'action': item['action']
                 }
        for mod in item['modality']:
            data_path = item[mod+'_path']
            if os.path.isdir(data_path):
                data_mod = self.read_dir(data_path)
            else:
                data_mod = np.load(data_path + '.npy')
            sample['input_'+mod] = data_mod

        return sample

def get_protocol_mapping(protocol):
    if protocol == 'protocol1':
        return protocol_1_mapping
    elif protocol == 'protocol2':
        return protocol_2_mapping
    else:
        return protocol_3_mapping
    
def make_datasets(args):
    mapping = get_protocol_mapping(args.protocol)
    database = MMFi_Database(args.dataset_path)
    config_dataset = decode_config(args)
    train_dataset = MMFi_Dataset(database, mapping, args.sequence_length, **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, mapping, args.sequence_length, **config_dataset['val_dataset'])
    test_dataset = MMFi_Dataset(database, mapping, args.sequence_length, **config_dataset['test_dataset'])

    return train_dataset, val_dataset, test_dataset

def make_dataloader(dataset, is_training, generator, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        pin_memory=True
    )
    return loader