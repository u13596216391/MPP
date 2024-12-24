import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EventFilter(nn.Module):
    def __init__(self, distance_threshold=2, max_events=100):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.max_events = max_events
    
    def nms_3d(self, points, scores):
        """3D非极大值抑制"""
        keep = torch.ones(len(points), dtype=torch.bool, device=points.device)
        points_float = points.float()
        
        for i in range(len(points)):
            if not keep[i]:
                continue
            
            cur_point = points_float[i]
            other_points = points_float[i+1:]
            
            if len(other_points) == 0:
                continue
                
            distances = torch.sqrt(((cur_point - other_points) ** 2).sum(dim=1))
            keep[i+1:][distances < self.distance_threshold] = False
            
        return keep

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        filtered = []
        
        for b in range(batch_size):
            seq_filtered = []
            for t in range(seq_len):
                energy = x[b, t, 0]
                magnitude = x[b, t, 1]
                points = torch.nonzero(energy)
                
                if len(points) == 0:
                    seq_filtered.append(x[b, t].unsqueeze(0))
                    continue
                
                scores = energy[points[:, 0], points[:, 1], points[:, 2]]
                sorted_idx = torch.argsort(scores, descending=True)
                points = points[sorted_idx]
                scores = scores[sorted_idx]
                
                keep = self.nms_3d(points, scores)
                
                # 限制每天最大事件数
                if keep.sum() > self.max_events:
                    keep[self.max_events:] = False
                
                filtered_map = torch.zeros_like(x[b, t])
                kept_points = points[keep]
                for p in kept_points:
                    filtered_map[0][p[0], p[1], p[2]] = energy[p[0], p[1], p[2]]
                    filtered_map[1][p[0], p[1], p[2]] = magnitude[p[0], p[1], p[2]]
                
                seq_filtered.append(filtered_map.unsqueeze(0))
            
            filtered.append(torch.cat(seq_filtered, dim=0).unsqueeze(0))
        
        return torch.cat(filtered, dim=0)

class SpatioTemporalNet(nn.Module):
    def __init__(self, spatial_size=(60, 15, 5), hidden_dim=128, num_layers=2, 
             dropout=0.1, input_seq_len=5, output_seq_len=3):
        super().__init__()
        self.spatial_size = spatial_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        
        # 编码器
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU()
            )
        ])
        
        # 特征压缩 - 修改输出通道为32以匹配解码器
        self.compressor = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=1),  # 改为32通道
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 4, 2))
        )
        
        # 自动计算LSTM输入维度
        with torch.no_grad():
            x = torch.zeros(1, 2, *spatial_size)
            for layer in self.encoder:
                x = layer(x)
            x = self.compressor(x)
            self.lstm_input_size = int(np.prod(x.shape[1:]))
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, self.lstm_input_size)
        
        # 解码器 - 修改通道数以匹配
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose3d(16, 2, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        ])
        
        self.event_filter = EventFilter(distance_threshold=2, max_events=100)
    
    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(-1, *x.shape[2:])
        
        # 编码
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        
        # 特征压缩
        x = self.compressor(x)
        compressed_shape = x.shape[1:]
        
        # LSTM处理
        features = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        x = self.fc(lstm_out)
        
        # 解码
        x = x.view(-1, *compressed_shape)  # 确保形状正确
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        
        # 确保输出维度正确
        if x.shape[-3:] != self.spatial_size:
            x = F.interpolate(x, size=self.spatial_size, mode='trilinear', align_corners=True)
        
        # 重塑输出并应用事件过滤
        x = x.view(batch_size, -1, *x.shape[1:])
        output = x[:, :self.output_seq_len]
        filtered_output = self.event_filter(output)
        
        return filtered_output