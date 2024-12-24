import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler
import os

class MicroseismicDataset(Dataset):
    def __init__(self, data_path, sequence_length=5, prediction_days=3, spatial_size=(60, 15, 5)):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.spatial_size = spatial_size
        
        # 读取数据
        self.data = pd.read_csv(data_path)
        
        # 确保必要的列存在
        required_columns = ['date', 'time', 'X', 'Y', 'Z', 'energy', 'magnitude']
        if not all(col in self.data.columns for col in required_columns):
            raise KeyError(f"Dataset must contain columns: {required_columns}")
        
        # 时间处理
        self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
        self.data = self.data.sort_values(by='datetime')
        
        # 特征缩放
        self.scalers = {
            'X': MinMaxScaler(),
            'Y': MinMaxScaler(),
            'Z': MinMaxScaler(),
            'energy': MinMaxScaler(),
            'magnitude': MinMaxScaler()
        }
        
        for feature in self.scalers:
            self.data[feature] = self.scalers[feature].fit_transform(
                self.data[feature].values.reshape(-1, 1)
            )
        
        # 创建时空序列
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        sequences = []
        dates = self.data['datetime'].dt.date.unique()
        
        for i in range(len(dates) - self.sequence_length - self.prediction_days + 1):
            # 输入序列
            input_dates = dates[i:i + self.sequence_length]
            input_data = [self._create_daily_grid(date) for date in input_dates]
            
            # 目标序列
            target_dates = dates[i + self.sequence_length:i + self.sequence_length + self.prediction_days]
            target_data = [self._create_daily_grid(date) for date in target_dates]
            
            sequences.append((input_data, target_data))
            
        return sequences
    
    def _create_daily_grid(self, date):
        """优化的网格划分函数"""
        day_data = self.data[self.data['datetime'].dt.date == date]
        grid = np.zeros((2, *self.spatial_size))  # [2, D, H, W]
        
        if len(day_data) == 0:
            return grid
            
        # 计算实际坐标到网格索引的映射
        x_min, x_max = 517100, 518100
        y_min, y_max = 4394700, 4395700
        z_min, z_max = 840, 940
        
        x_idx = ((day_data['X'] - x_min) / (x_max - x_min) * (self.spatial_size[0] - 1)).astype(int)
        y_idx = ((day_data['Y'] - y_min) / (y_max - y_min) * (self.spatial_size[1] - 1)).astype(int)
        z_idx = ((day_data['Z'] - z_min) / (z_max - z_min) * (self.spatial_size[2] - 1)).astype(int)
        
        # 限制索引范围
        x_idx = np.clip(x_idx, 0, self.spatial_size[0]-1)
        y_idx = np.clip(y_idx, 0, self.spatial_size[1]-1)
        z_idx = np.clip(z_idx, 0, self.spatial_size[2]-1)
        
        # 合并相同网格的事件
        for i in range(len(day_data)):
            grid[0, x_idx.iloc[i], y_idx.iloc[i], z_idx.iloc[i]] = max(
                grid[0, x_idx.iloc[i], y_idx.iloc[i], z_idx.iloc[i]],
                day_data['energy'].iloc[i]
            )
            grid[1, x_idx.iloc[i], y_idx.iloc[i], z_idx.iloc[i]] = max(
                grid[1, x_idx.iloc[i], y_idx.iloc[i], z_idx.iloc[i]],
                day_data['magnitude'].iloc[i]
            )
        
        return grid
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        # 使用numpy.stack替代list转换
        input_tensor = torch.from_numpy(np.stack(input_seq, axis=0)).float()
        target_tensor = torch.from_numpy(np.stack(target_seq, axis=0)).float()
        
        return input_tensor, target_tensor

def split_and_save_data(data_path, output_dir, train_ratio=0.7, val_ratio=0.2):
    # 读取数据
    data = pd.read_csv(data_path)
    
    # 确保必要的列存在
    required_columns = ['date', 'time', 'X', 'Y', 'Z', 'energy', 'magnitude']
    if not all(col in data.columns for col in required_columns):
        raise KeyError(f"Dataset must contain columns: {required_columns}")
    
    # 处理时间
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data = data.sort_values(by='datetime')
    
    # 按日期分割
    unique_dates = data['date'].unique()
    total_days = len(unique_dates)
    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)
    
    # 分割数据集
    train_dates = unique_dates[:train_days]
    val_dates = unique_dates[train_days:train_days + val_days]
    test_dates = unique_dates[train_days + val_days:]
    
    # 创建数据子集
    train_data = data[data['date'].isin(train_dates)]
    val_data = data[data['date'].isin(val_dates)]
    test_data = data[data['date'].isin(test_dates)]
    
    # 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)