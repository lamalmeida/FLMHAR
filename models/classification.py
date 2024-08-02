from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, encoder, classification_head):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.classification_head = classification_head

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classification_head(encoded)
        return output