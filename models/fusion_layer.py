import torch
from torch import nn

class FusionLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FusionLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) 
        self.bn1 = nn.BatchNorm1d(64)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(64, num_classes) 
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x