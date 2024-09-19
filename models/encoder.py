import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=297):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SensorSpecificEncoder(nn.Module):
    def __init__(self, input_shape, d_model):
        super(SensorSpecificEncoder, self).__init__()
        input_dim = input_shape[-1] * input_shape[-2]  
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, d_model)
        self.bn1 = nn.BatchNorm1d(input_shape[0])
        self.bn2 = nn.BatchNorm1d(input_shape[0])
        self.bn3 = nn.BatchNorm1d(input_shape[0])
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        
    def forward(self, x):
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 560)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout after activation
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)  # Apply dropout after activation
        x = torch.relu(self.bn3(self.fc3(x)))
        
        return x

class SensorEncoder(nn.Module):
    def __init__(self, mod, input_shape, num_heads, d_model, num_transformer_blocks, dropout=0):
        super(SensorEncoder, self).__init__()
        self.sensor_encoder = SensorSpecificEncoder(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=297)
        self.transformer_blocks = nn.ModuleList([
            self.build_transformer_block(num_heads, d_model, dropout) for _ in range(num_transformer_blocks)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_dim = d_model

    def build_transformer_block(self, num_heads, d_model, dropout):
        return nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Sequential(
                nn.Linear(d_model, d_model*4),
                nn.ReLU(),
                nn.Dropout(dropout), 
                nn.Linear(d_model*4, d_model)
            ),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        ])

    def forward(self, inputs, mask=None):
        x = self.sensor_encoder(inputs)
        x = self.pos_encoder(x) 

        for attention, norm1, drop1, ffn, norm2, drop2 in self.transformer_blocks:
            attn_output, _ = attention(x, x, x, attn_mask=mask)
            attn_output = drop1(attn_output)
            out1 = norm1(x + attn_output)
            ffn_output = ffn(out1)
            ffn_output = drop2(ffn_output)
            x = norm2(out1 + ffn_output)

        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return x
    
def make_encoders(mod, input_shape, args, pre_trained_path=None):
    model = SensorEncoder(mod, input_shape, args.num_heads, args.d_model, args.num_transformer_blocks, args.dropout)

    if pre_trained_path:
        model.load_state_dict(torch.load(pre_trained_path))

    return model