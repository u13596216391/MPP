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

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(hidden_states * attention_weights, dim=1)
        return attended

class SpatioTemporalNet(nn.Module):
    def __init__(self, spatial_size=(60, 15, 5), hidden_dim=128, num_layers=2,
                 dropout=0.1, input_seq_len=5, output_seq_len=3):
        super().__init__()
        self.spatial_size = spatial_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        
        # 计算每个维度的下采样次数
        min_dim = min(spatial_size)
        self.n_down = int(np.log2(min_dim)) - 2  # 确保最小维度至少为4
        
        # 编码器层
        self.encoder = self._build_encoder()
        
        # 计算压缩后的特征尺寸
        with torch.no_grad():
            x = torch.zeros(1, 2, *spatial_size)
            for enc in self.encoder:
                x = enc(x)
            self.compressed_shape = x.shape[1:]  # 保存压缩后的形状
            self.feature_dim = int(np.prod(x.shape[1:]))
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = Attention(hidden_dim)
        self.pre_decoder = nn.Linear(hidden_dim, self.feature_dim)
        
        # 解码器层
        self.decoder = self._build_decoder()
        self.event_filter = EventFilter()
        
    def _build_encoder(self):
        layers = []
        in_channels = 2
        out_channels = 32
        
        for i in range(self.n_down):
            layers.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.MaxPool3d(2)
            ))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)
            
        return nn.ModuleList(layers)
        
    def _build_decoder(self):
        layers = []
        in_channels = self.compressed_shape[0]  # 使用压缩后的通道数
        
        for i in range(self.n_down):
            out_channels = max(in_channels // 2, 32)
            if i == self.n_down - 1:
                out_channels = 2
                
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU() if i < self.n_down - 1 else nn.Sigmoid()
            ))
            in_channels = out_channels
            
        return nn.ModuleList(layers)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 编码序列
        encoded_seqs = []
        for t in range(self.input_seq_len):
            curr_input = x[:, t]
            # 编码过程
            for enc in self.encoder:
                curr_input = enc(curr_input)
            encoded_seqs.append(curr_input.flatten(1))
        
        encoded = torch.stack(encoded_seqs, dim=1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(encoded)
        
        # 生成预测序列
        outputs = []
        curr_input = self.attention(lstm_out)
        
        for _ in range(self.output_seq_len):
            # 解码一个时间步
            features = self.pre_decoder(curr_input)
            features = features.view(batch_size, *self.compressed_shape)
            
            # 解码
            for dec in self.decoder:
                features = dec(features)
            
            # 确保输出尺寸匹配
            if features.shape[-3:] != self.spatial_size:
                features = F.interpolate(
                    features, 
                    size=self.spatial_size,
                    mode='trilinear',
                    align_corners=True
                )
            
            outputs.append(features)
            curr_input = self.attention(lstm_out)
        
        # 组合输出并过滤
        output = torch.stack(outputs, dim=1)
        filtered_output = self.event_filter(output)
        
        return filtered_output
class LSTMCNNModel(nn.Module):
    def __init__(self, spatial_size=(60, 15, 5), hidden_dim=128, num_layers=2,
                 dropout=0.1, input_seq_len=5, output_seq_len=3):
        super().__init__()
        self.spatial_size = spatial_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # CNN编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # 计算flatten后的特征维度
        with torch.no_grad():
            x = torch.zeros(1, 2, *spatial_size)
            x = self.encoder(x)
            self.feature_dim = x.numel()
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.event_filter = EventFilter()
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 编码序列
        encoded_seqs = []
        for t in range(self.input_seq_len):
            features = self.encoder(x[:, t])
            encoded_seqs.append(features.view(batch_size, -1))
        
        # LSTM处理
        lstm_in = torch.stack(encoded_seqs, dim=1)
        lstm_out, _ = self.lstm(lstm_in)
        
        # 解码预测序列
        outputs = []
        output = torch.stack(outputs, dim=1)
        filtered_output = self.event_filter(output)
        return filtered_output

class UNet3D(nn.Module):
    def __init__(self, spatial_size=(60, 15, 5), input_channels=2, base_filters=16,
                 input_seq_len=5, output_seq_len=3):  # 添加序列长度参数
        super().__init__()
        self.spatial_size = spatial_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # 下采样路径
        self.enc1 = self._make_layer(input_channels, base_filters)
        self.enc2 = self._make_layer(base_filters, base_filters*2)
        self.enc3 = self._make_layer(base_filters*2, base_filters*4)
        
        # 上采样路径
        self.dec3 = self._make_layer(base_filters*8, base_filters*4)
        self.dec2 = self._make_layer(base_filters*6, base_filters*2)
        self.dec1 = self._make_layer(base_filters*3, base_filters)
        
        self.final = nn.Conv3d(base_filters, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.event_filter = EventFilter()
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        
        # 处理每个输入时间步
        for t in range(self.output_seq_len):
            # 使用最后一个输入进行预测
            features = x[:, -1]
            # 编码-解码过程
            enc1 = self.enc1(features)
            enc2 = self.enc2(F.max_pool3d(enc1, 2))
            enc3 = self.enc3(F.max_pool3d(enc2, 2))
            
            dec3 = self.dec3(torch.cat([F.interpolate(enc3, size=enc2.shape[2:]), enc2], 1))
            dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc1.shape[2:]), enc1], 1))
            dec1 = self.dec1(dec2)
            
            output = self.sigmoid(self.final(dec1))
            outputs.append(output)
        
        # 组合输出并过滤
        output = torch.stack(outputs, dim=1)
        filtered_output = self.event_filter(output)
        
        return filtered_output