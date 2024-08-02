import argparse

def get_args():
    parser = argparse.ArgumentParser(description="MMFi Dataset Training")
    
    parser.add_argument('--dataset_path', type=str, default='dataset/', help='Path to the dataset')

    parser.add_argument('--modalities', type=str, default='mmwave', help='Sensor modalities to use (rgb|lidar|mmwave|wifi-csi)')
    parser.add_argument('--protocol', type=str, default='protocol3', help='Protocol to use')
    parser.add_argument('--sequence_length', type=int, default=297, help='Number of frames in the sequence (must be a division of 297)')
    parser.add_argument('--val_ratio', type=int, default=15, help='Percentage of datapoints for validation dataset')
    parser.add_argument('--test_ratio', type=int, default=15, help='Percentage of datapoints for test dataset')

    # Encoder arguments
    parser.add_argument('--num_heads', type=int, default=4, help='Heads of the transformer encoders')
    parser.add_argument('--d_model', type=int, default=128, help='Hidden dimension of transformer encoder')
    parser.add_argument('--num_transformer_blocks', type=int, default=4, help='Number of transformer blcosk for encoder')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for transformer encoders')

    # Encoder training arguments
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--contrastive_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--contrastive_learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['classification', 'decision_fusion', 'feature_fusion'], help='Model to train')
    
    # Classification training arguments
    parser.add_argument('--classification_learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--classification_epochs', type=int, default=60, help='Number of training epochs')

    # Federated Learning arguments
    # parser.add_argument('--num_clients', type=int, default=10, help='Number of clients in federated learning')
    # parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds')
    
    args = parser.parse_args()
    return args