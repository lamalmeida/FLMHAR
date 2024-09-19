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
    
# class SensorSpecificEncoder(nn.Module):
#     def __init__(self, input_shape, d_model):
#         super(SensorSpecificEncoder, self).__init__()
#         self.depthwise_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, groups=1)
#         self.pointwise_conv1 = nn.Conv2d(64, 64, kernel_size=1)
#         self.depthwise_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64)
#         self.pointwise_conv2 = nn.Conv2d(128, 128, kernel_size=1)
#         self.depthwise_conv3 = nn.Conv2d(128, d_model, kernel_size=3, padding=1, groups=128)
#         self.pointwise_conv3 = nn.Conv2d(d_model, d_model, kernel_size=1)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         batch_size, num_frames, num_points, feature_dim = x.shape
#         x = x.view(-1, feature_dim, num_points)
#         x = x.unsqueeze(1)
#         x = self.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
#         x = self.pool(x)
#         x = self.relu(self.pointwise_conv2(self.depthwise_conv2(x)))
#         x = self.pool(x)
#         x = self.relu(self.pointwise_conv3(self.depthwise_conv3(x)))
#         x = x.view(batch_size, num_frames, -1)
        
#         return x

class SensorSpecificEncoder(nn.Module):
    def __init__(self, input_shape, d_model):
        super(SensorSpecificEncoder, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # self.pool1 = nn.AdaptiveAvgPool2d((5, 28))
        # self.conv2 = nn.Conv2d(64, d_model, kernel_size=3, padding=1)
        # self.pool2 = nn.AdaptiveAvgPool2d((1, 1))  # Pool after second conv layer
        # self.relu = nn.ReLU()
        input_dim = input_shape[-2]
        self.mlp1 = nn.Conv1d(input_dim, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, d_model, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        # batch_size, num_frames, num_points, feature_dim = x.shape
        # x = x.view(-1, 1, feature_dim, num_points)
        # x = self.relu(self.conv1(x)) # (19008, 64, 5, 112)
        # x = self.pool1(x)  # (19008, 64, 3, 56)
        
        # x = self.relu(self.conv2(x))  # (19008, d_model, 5, 56)
        # x = self.pool2(x)  # (19008, d_model, 1, 1)
        
        # x = x.view(batch_size, num_frames, -1)  # (64, 297, d_model)
        x = torch.relu(self.bn1(self.mlp1(x)))
        x = torch.relu(self.bn2(self.mlp2(x)))
        x = torch.relu(self.bn3(self.mlp3(x)))
        x = torch.mean(x, 2)
        
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
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        ])

    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        encoded_outputs = []

        for i in range(batch_size):
            x = self.sensor_encoder(inputs[i])
            encoded_outputs.append(x)
        
        x = torch.stack(encoded_outputs, dim=0)
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