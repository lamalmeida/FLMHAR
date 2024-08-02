import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from torch import nn
import time
import numpy as np
import gc
    
def train_step(encoder, classification_head, optimizer, data, labels):
    encoder.eval()
    classification_head.train()
    optimizer.zero_grad()

    with torch.no_grad():
        features = encoder(data)
    outputs = classification_head(features)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(encoder, classification_head, data, labels, metrics):
    encoder.eval()
    classification_head.eval()

    with torch.no_grad():
        outputs = classification_head(encoder(data))

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    _, preds = torch.max(outputs, 1)

    # Update metrics
    metrics.update(preds, labels)

    return loss.item()

def train_classification(encoder, classification_head, optimizer, train_loader, val_loader, epochs, args):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = []
        for data in train_loader:
            data['input_' + args.modalities] = data['input_' + args.modalities].to(args.device)
            data['action'] = data['action'].to(args.device)
            total_loss.append(train_step(encoder, classification_head, optimizer, data['input_' + args.modalities], data['action']))
                
        print(f'[{args.modalities:<10}] Epoch {epoch + 1:>2}, Loss: {np.mean(total_loss):.4f} ({time.time() - start_time:.4f})')

        start_time = time.time()
        val_loss = []
        val_metrics = MetricCollection([
                Accuracy(num_classes=classification_head.fc2.out_features, average='weighted', task='multiclass'),
                Precision(num_classes=classification_head.fc2.out_features, average='weighted', task='multiclass'),
                Recall(num_classes=classification_head.fc2.out_features, average='weighted', task='multiclass'),
                F1Score(num_classes=classification_head.fc2.out_features, average='weighted', task='multiclass')
            ]).to(args.device) 
        for val_data in val_loader:
            val_data['input_' + args.modalities] = val_data['input_' + args.modalities].to(args.device)
            val_data['action'] = val_data['action'].to(args.device)
            val_loss.append(validation_step(encoder, classification_head, val_data['input_' + args.modalities], val_data['action'], val_metrics))
        
        metrics = val_metrics.compute()
        print(f'[{args.modalities:<10}] Epoch {epoch + 1:>2}, Loss: {np.mean(val_loss):.4f}, Accuracy: {metrics["MulticlassAccuracy"].item():.4f}, F1: {metrics["MulticlassF1Score"].item():.4f}, Precision: {metrics["MulticlassPrecision"].item():.4f}, Recall: {metrics["MulticlassRecall"].item():.4f} ({time.time() - start_time:.4f})')
        torch.save(classification_head.state_dict(), f'ckpt/classifiers/{args.modalities}_epoch_{epoch + 1}.pth')

        scheduler.step()

        gc.collect()
        torch.mps.empty_cache()