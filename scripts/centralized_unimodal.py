import torch
import time

from torchsummary import summary

from mmfi_lib.mmfi import make_datasets, make_dataloader
from models.encoder import make_encoders
from models.classification import ClassificationHead
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

def create_dataloaders(args):
    train_dataset, val_dataset, test_dataset = make_datasets(args)

    rng_generator = torch.manual_seed(0)
    train_loader = make_dataloader(train_dataset, True, rng_generator, args.batch_size)
    val_loader = make_dataloader(val_dataset, False, rng_generator, args.batch_size)
    test_loader = make_dataloader(test_dataset, False, rng_generator, args.batch_size)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args.num_classes = num_classes[args.protocol]

    all_mods = args.modalities.split('|')
    for mod in all_mods:
        start_time = time.time()
        args.modalities = mod
        train_loader, val_loader, test_loader = create_dataloaders(args)
        print(f'Created {mod} data loaders in {time.time() - start_time:.2f}s')

        encoder = make_encoders(
                mod,
                input_shapes[mod],
                args,
                pre_trained_path='ckpt/encoders/intermediate/rgb_epoch_29_valloss_1.26.pth'
            ).to(args.device)
        # summary(encoders[mod], input_size=input_shapes[mod])

        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.contrastive_learning_rate)

        # train_contrastive(encoder, optimizer, train_loader, val_loader, epochs=args.contrastive_epochs, args=args)

        classification_head = ClassificationHead(args.d_model, args.num_classes).to(args.device)

        optimizer = torch.optim.Adam(classification_head.parameters(), lr=args.classification_learning_rate)

        train_classification(encoder, classification_head, optimizer, train_loader, val_loader, epochs=args.classification_epochs, args=args)




