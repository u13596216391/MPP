import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime, timedelta
import os
import torch
from core.data_processing import MicroseismicDataset
class DataVisualizer:
    def __init__(self, spatial_size=(60, 15, 5)):
        self.spatial_size = spatial_size
        self.coord_ranges = {
            'X': (517100, 518100),
            'Y': (4394700, 4395700),
            'Z': (840, 940)
        }
    
    def grid_to_points(self, grid_data):
        """将网格数据转换为点数据"""
        points = []
        for t in range(grid_data.shape[0]):
            positions = np.nonzero(grid_data[t, 0])
            for idx in range(len(positions[0])):
                x = self.coord_ranges['X'][0] + (positions[0][idx] / self.spatial_size[0]) * (self.coord_ranges['X'][1] - self.coord_ranges['X'][0])
                y = self.coord_ranges['Y'][0] + (positions[1][idx] / self.spatial_size[1]) * (self.coord_ranges['Y'][1] - self.coord_ranges['Y'][0])
                z = self.coord_ranges['Z'][0] + (positions[2][idx] / self.spatial_size[2]) * (self.coord_ranges['Z'][1] - self.coord_ranges['Z'][0])
                
                points.append({
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'energy': float(grid_data[t, 0, positions[0][idx], positions[1][idx], positions[2][idx]]),
                    'magnitude': float(grid_data[t, 1, positions[0][idx], positions[1][idx], positions[2][idx]])
                })
        return pd.DataFrame(points)


def merge_nearby_events(events_df, distance_threshold=50):
    """合并临近事件"""
    merged_events = []
    
    # 按日期分组处理
    for date, day_events in events_df.groupby('date'):
        processed = set()
        
        for idx1, row1 in day_events.iterrows():
            if idx1 in processed:
                continue
                
            # 找到当前点附近的所有点
            nearby_events = []
            for idx2, row2 in day_events.iterrows():
                if idx2 != idx1 and idx2 not in processed:
                    dist = np.sqrt(
                        (row1['X'] - row2['X'])**2 + 
                        (row1['Y'] - row2['Y'])**2 + 
                        (row1['Z'] - row2['Z'])**2
                    )
                    if dist < distance_threshold:
                        nearby_events.append(row2)
                        processed.add(idx2)
            
            # 如果有临近事件，合并它们
            if nearby_events:
                energies = [row1['energy']] + [e['energy'] for e in nearby_events]
                magnitudes = [row1['magnitude']] + [e['magnitude'] for e in nearby_events]
                merged_events.append({
                    'date': row1['date'],
                    'time': row1['time'],
                    'X': row1['X'],
                    'Y': row1['Y'],
                    'Z': row1['Z'],
                    'energy': max(energies),
                    'magnitude': max(magnitudes),
                    'type': row1['type']
                })
            else:
                merged_events.append(row1.to_dict())
    
    return pd.DataFrame(merged_events)
def convert_to_coords(data, timestamps=None, is_prediction=True):
    """转换网格数据为实际坐标"""
    coords_list = []
    
    # 坐标映射参数（与数据处理保持一致）
    x_min, x_max = 517100, 518100
    y_min, y_max = 4394700, 4395700
    z_min, z_max = 840, 940
    
    # 网格尺寸
    grid_size = data.shape[2:]  # [D, H, W]
    
    if timestamps is None:
        base_time = datetime.now()
        timestamps = [base_time + timedelta(days=i) for i in range(data.shape[0])]
    
    for t in range(data.shape[0]):
        positions = np.nonzero(data[t, 0])
        if len(positions[0]) == 0:
            continue
            
        for idx in range(len(positions[0])):
            # 将网格索引转换为实际坐标
            x = x_min + (positions[0][idx] / grid_size[0]) * (x_max - x_min)
            y = y_min + (positions[1][idx] / grid_size[1]) * (y_max - y_min)
            z = z_min + (positions[2][idx] / grid_size[2]) * (z_max - z_min)
            
            coords_list.append({
                'date': timestamps[t].strftime('%Y-%m-%d'),
                'time': timestamps[t].strftime('%H:%M:%S'),
                'X': x,
                'Y': y,
                'Z': z,
                'energy': float(data[t, 0, positions[0][idx], positions[1][idx], positions[2][idx]]),
                'magnitude': float(data[t, 1, positions[0][idx], positions[1][idx], positions[2][idx]]),
                'type': 'prediction' if is_prediction else 'target'
            })
    
    return coords_list


def visualize_prediction(input_seq, prediction, target, save_path=None):
    """改进的分离式可视化函数"""
    fig = plt.figure(figsize=(20, 10))
    
    # 创建两个子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 坐标范围
    x_range = (517100, 518100)
    y_range = (4394700, 4395700)
    z_range = (840, 940)
    
    # 绘制预测结果
    pred_points = np.where(prediction[0] > 0)
    if len(pred_points[0]) > 0:
        pred_df = pd.DataFrame({
            'X': [x_range[0] + (x/60)*(x_range[1]-x_range[0]) for x in pred_points[0]],
            'Y': [y_range[0] + (y/15)*(y_range[1]-y_range[0]) for y in pred_points[1]],
            'Z': [z_range[0] + (z/5)*(z_range[1]-z_range[0]) for z in pred_points[2]],
            'energy': prediction[0][pred_points],
            'magnitude': prediction[1][pred_points]
        })
        scatter1 = ax1.scatter(pred_df['X'], pred_df['Y'], pred_df['Z'],
                             c=pred_df['energy'], cmap='hot',
                             label='Prediction', s=50)
        plt.colorbar(scatter1, ax=ax1, label='Energy')
    else:
        pred_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'energy', 'magnitude'])
    
    # 绘制真实值
    target_points = np.where(target[0] > 0)
    if len(target_points[0]) > 0:
        target_df = pd.DataFrame({
            'X': [x_range[0] + (x/60)*(x_range[1]-x_range[0]) for x in target_points[0]],
            'Y': [y_range[0] + (y/15)*(y_range[1]-y_range[0]) for y in target_points[1]],
            'Z': [z_range[0] + (z/5)*(z_range[1]-z_range[0]) for z in target_points[2]],
            'energy': target[0][target_points],
            'magnitude': target[1][target_points]
        })
        scatter2 = ax2.scatter(target_df['X'], target_df['Y'], target_df['Z'],
                             c=target_df['energy'], cmap='hot',
                             label='Ground Truth', s=50)
        plt.colorbar(scatter2, ax=ax2, label='Energy')
    else:
        target_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'energy', 'magnitude'])

    # 设置两个子图的属性
    for ax, title in zip([ax1, ax2], ['Prediction', 'Ground Truth']):
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')
        ax.set_title(title)
        ax.grid(True)
        
        # 设置相同的视角
        ax.view_init(elev=20, azim=45)
        
        # 设置相同的坐标范围
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'pred_df': pred_df,
        'target_df': target_df,
        'figure': fig
    }
def visualize_results(val_data_path, test_data_path, model, device, save_dir):
    """可视化验证集和测试集的预测结果"""
    # 加载数据集
    val_dataset = MicroseismicDataset(val_data_path)
    test_dataset = MicroseismicDataset(test_data_path)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    def generate_timestamps(num_days):
        base_time = datetime.now()
        return [base_time + timedelta(days=i) for i in range(num_days)]
    
    # 可视化验证集
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        timestamps = generate_timestamps(output.shape[1])
        visualize_prediction(
            data[0].cpu().numpy(),
            output[0].cpu().numpy(),
            target[0].cpu().numpy(),
            timestamps=timestamps,
            save_path=os.path.join(save_dir, 'val_predictions.png')
        )
        break
    # 可视化测试集
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        timestamps = generate_timestamps(output.shape[1])
        visualize_prediction(
            data[0].cpu().numpy(),
            output[0].cpu().numpy(),
            target[0].cpu().numpy(),
            timestamps=timestamps,
            save_path=os.path.join(save_dir, 'test_predictions.png')
        )
        break