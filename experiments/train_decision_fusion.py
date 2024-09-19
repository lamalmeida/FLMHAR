import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from torch import nn
import torch.nn.functional as F
import time
from collections import defaultdict
import numpy as np
import gc

class MaskedAveragePooling1d(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling1d, self).__init__()

    def forward(self, x):
        mask = (x != 0).float()
        sum_values = (x * mask).sum(dim=-1, keepdim=True)
        count_values = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Avoid division by zero
        
        return sum_values / count_values
    
def train_step(encoders, classification_heads, final_classifier, optimizer, data, labels, device):
    for mod in encoders.keys():
        encoders[mod].eval()
        classification_heads[mod].train()
    final_classifier.train()

    optimizer.zero_grad()

    decisions = []
    for i, mod in enumerate(encoders.keys()):
        train_data = data[i]  
        
        with torch.no_grad():
            encoded_data = encoders[mod](train_data) 
        
        decision = classification_heads[mod](encoded_data)  
        decisions.append(decision)
    
    # decisions = torch.mean(torch.stack(decisions), dim=0)
    decisions = torch.cat(decisions, dim=1)
    
    outputs = final_classifier(decisions) 

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(encoders, classification_heads, final_classifier, data, labels, val_metrics, device):
    for mod in encoders.keys():
        encoders[mod].eval()
        classification_heads[mod].eval()
    final_classifier.eval()

    decisions = []
    
    with torch.no_grad():
        for i, mod in enumerate(encoders.keys()):
            val_data = data[i] 
            
            encoded_data = encoders[mod](val_data)
            decision = classification_heads[mod](encoded_data)
            decisions.append(decision)

        # decisions = torch.mean(torch.stack(decisions), dim=0)
        decisions = torch.cat(decisions, dim=1)
        
        outputs = final_classifier(decisions)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    _, preds = torch.max(outputs, 1)

    val_metrics.update(preds, labels)

    return loss.item()

def train_decision_fusion(encoders, classification_heads, final_classifier, optimizer, train_loader, val_loader, args):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 
    max_f1 = 0

    for epoch in range(args.classification_epochs):
        start_time = time.time()
        total_loss = []

        modality_keys = list(train_loader.keys())

        iterators = {mod: iter(train_loader[mod]) for mod in modality_keys}

        while True:
            train_data = []
            train_label = None

            try:
                for mod in modality_keys:
                    data = next(iterators[mod])

                    if train_label is None:
                        train_label = data[1].to(args.device)  

                    train_data.append(data[0].to(args.device))  

                total_loss.append(train_step(encoders, classification_heads, final_classifier, optimizer, train_data, train_label, args.device))

            except StopIteration:
                break
        print(f'Epoch {epoch + 1:>2}, Loss: {np.mean(total_loss):.4f} ({time.time() - start_time:.4f})')

        if (epoch + 1) % 1 == 0:
            start_time = time.time()
            val_loss = []
            
            val_metrics = MetricCollection([
                Accuracy(num_classes=args.num_classes, average='weighted', task='multiclass'),
                Precision(num_classes=args.num_classes, average='weighted', task='multiclass'),
                Recall(num_classes=args.num_classes, average='weighted', task='multiclass'),
                F1Score(num_classes=args.num_classes, average='weighted', task='multiclass')
            ]).to(args.device) 

            modality_keys = list(val_loader.keys())

            iterators = {mod: iter(val_loader[mod]) for mod in modality_keys}

            while True:
                val_data = []
                val_label = None

                try:
                    for mod in modality_keys:
                        data = next(iterators[mod])

                        if val_label is None:
                            val_label = data[1].to(args.device)  

                        val_data.append(data[0].to(args.device)) 

                    val_loss.append(validation_step(encoders, classification_heads, final_classifier, val_data, val_label, val_metrics, args.device))

                except StopIteration:
                    break

            metrics = val_metrics.compute()
            
            print(f'Epoch {epoch + 1:>2}, Loss: {np.mean(val_loss):.4f}, '
                f'Accuracy: {metrics["MulticlassAccuracy"].item():.4f}, '
                f'F1: {metrics["MulticlassF1Score"].item():.4f}, '
                f'Precision: {metrics["MulticlassPrecision"].item():.4f}, '
                f'Recall: {metrics["MulticlassRecall"].item():.4f} '
                f'({time.time() - start_time:.4f})')
            if max_f1 < metrics["MulticlassF1Score"].item():
                max_f1 = max(max_f1, metrics["MulticlassF1Score"].item())
                file = f'ckpt/classifiers/epoch_{epoch + 1}.pth'
                torch.save({
                    'classification_heads': [classification_heads[head].state_dict() for head in classification_heads], 
                    'final_classifier': final_classifier.state_dict()
                }, file)

        scheduler.step()

        gc.collect()
        torch.mps.empty_cache()

        checkpoint = torch.load(file)
        for i, head in enumerate(classification_heads):
            classification_heads[head].load_state_dict(checkpoint['classification_heads'][i])
        final_classifier.load_state_dict(checkpoint['final_classifier'])