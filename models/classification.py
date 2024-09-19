from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) 
        self.bn1 = nn.BatchNorm1d(128)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(128, 64)  
        self.bn2 = nn.BatchNorm1d(64) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(64, num_classes) 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
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