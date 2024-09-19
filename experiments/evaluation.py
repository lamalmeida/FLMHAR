import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_unimodal(encoder, classification_head, test_loader, device=torch.device('mps')):
    encoder.eval()
    classification_head.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch[0], batch[1]
            data, labels = data.to(device), labels.to(device)

            encoded_data = encoder(data)
            outputs = classification_head(encoded_data)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Print results
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall: {recall*100:.2f}")
    print(f"F1 Score: {f1*100:.2f}")

    # Plot confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.show()

def evaluate_multimodal_decision(encoders, classification_heads, final_classifier, test_loader, device=torch.device('mps')):
    val_metrics = MetricCollection([
        Accuracy(num_classes=27, average='weighted', task='multiclass'),
        Precision(num_classes=27, average='weighted', task='multiclass'),
        Recall(num_classes=27, average='weighted', task='multiclass'),
        F1Score(num_classes=27, average='weighted', task='multiclass')
    ]).to(device) 

    modality_keys = list(test_loader.keys())

    iterators = {mod: iter(test_loader[mod]) for mod in modality_keys}

    while True:
        val_data = []
        val_label = None

        try:
            for mod in modality_keys:
                data = next(iterators[mod])

                if val_label is None:
                    val_label = data[1].to(device)  

                val_data.append(data[0].to(device)) 

            test_step_decision(encoders, classification_heads, final_classifier, val_data, val_label, val_metrics)

        except StopIteration:
            break

    metrics = val_metrics.compute()
    
    print(f'Accuracy: {metrics["MulticlassAccuracy"].item():.4f}, '
        f'F1: {metrics["MulticlassF1Score"].item():.4f}, '
        f'Precision: {metrics["MulticlassPrecision"].item():.4f}, '
        f'Recall: {metrics["MulticlassRecall"].item():.4f} ')

def test_step_decision(encoders, classification_heads, final_classifier, data, labels, val_metrics):
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

    _, preds = torch.max(outputs, 1)

    val_metrics.update(preds, labels)

def evaluate_multimodal_features(encoders, final_classifier, test_loader, device=torch.device('mps')):
    val_metrics = MetricCollection([
        Accuracy(num_classes=27, average='weighted', task='multiclass'),
        Precision(num_classes=27, average='weighted', task='multiclass'),
        Recall(num_classes=27, average='weighted', task='multiclass'),
        F1Score(num_classes=27, average='weighted', task='multiclass')
    ]).to(device) 

    modality_keys = list(test_loader.keys())

    iterators = {mod: iter(test_loader[mod]) for mod in modality_keys}

    while True:
        val_data = []
        val_label = None

        try:
            for mod in modality_keys:
                data = next(iterators[mod])

                if val_label is None:
                    val_label = data[1].to(device)  

                val_data.append(data[0].to(device)) 

            test_step_features(encoders, final_classifier, val_data, val_label, val_metrics)

        except StopIteration:
            break

    metrics = val_metrics.compute()
    
    print(f'Accuracy: {metrics["MulticlassAccuracy"].item():.4f}, '
        f'F1: {metrics["MulticlassF1Score"].item():.4f}, '
        f'Precision: {metrics["MulticlassPrecision"].item():.4f}, '
        f'Recall: {metrics["MulticlassRecall"].item():.4f} ')

def test_step_features(encoders, final_classifier, data, labels, val_metrics):
    for mod in encoders.keys():
        encoders[mod].eval()
    final_classifier.eval()

    decisions = []
    
    with torch.no_grad():
        for i, mod in enumerate(encoders.keys()):
            val_data = data[i] 
            
            encoded_data = encoders[mod](val_data)
            decisions.append(encoded_data)

        # decisions = torch.mean(torch.stack(decisions), dim=0)
        decisions = torch.cat(decisions, dim=1)
        
        outputs = final_classifier(decisions)

    _, preds = torch.max(outputs, 1)

    val_metrics.update(preds, labels)