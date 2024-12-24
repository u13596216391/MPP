import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime, timedelta
import os

def visualize_prediction(input_seq, prediction, target, save_path=None, spatial_size=(180, 30, 10)):
    """
    可视化3D预测结果并保存为CSV格式
    Args:
        input_seq: [seq_len, channels, D, H, W]
        prediction: [seq_len, channels, D, H, W] 
        target: [seq_len, channels, D, H, W]
        save_path: 保存路径
        spatial_size: 空间尺寸
    """
    # 创建outputs目录
    output_dir = os.path.join(os.path.dirname(save_path), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取最后一个时间点
    last_time = datetime.now()
    
    # 转换网格索引到实际坐标
    x_min, x_max = 517100, 518100  # 根据train.csv的实际范围设置
    y_min, y_max = 4394700, 4395700
    z_min, z_max = 840, 940

    def grid_to_coords(indices, grid_values):
        x = x_min + (indices[0] / (spatial_size[0] - 1)) * (x_max - x_min)
        y = y_min + (indices[1] / (spatial_size[1] - 1)) * (y_max - y_min)
        z = z_min + (indices[2] / (spatial_size[2] - 1)) * (z_max - z_min)
        return x, y, z, grid_values

    # 转换预测和目标数据到坐标格式
    def convert_to_coords(data, is_prediction=True):
        coords_list = []
        for t in range(data.shape[0]):
            energy_grid = data[t, 0]  # 能量通道
            magnitude_grid = data[t, 1]  # 震级通道
            
            # 找到非零值的位置
            nonzero = np.nonzero(energy_grid)
            for idx in range(len(nonzero[0])):
                x, y, z, energy = grid_to_coords(
                    (nonzero[0][idx], nonzero[1][idx], nonzero[2][idx]),
                    energy_grid[nonzero[0][idx], nonzero[1][idx], nonzero[2][idx]]
                )
                magnitude = magnitude_grid[nonzero[0][idx], nonzero[1][idx], nonzero[2][idx]]
                time = last_time + timedelta(days=t)
                coords_list.append([
                    time.strftime('%Y/%m/%d'),
                    time.strftime('%H:%M:%S'),
                    x, y, z, energy, magnitude,
                    'Predicted' if is_prediction else 'Actual'
                ])
        return coords_list

    # 转换预测值和实际值
    pred_coords = convert_to_coords(prediction, True)
    target_coords = convert_to_coords(target, False)
    
    # 合并数据并创建DataFrame
    all_coords = pred_coords + target_coords
    df = pd.DataFrame(all_coords, columns=[
        'date', 'time', 'X', 'Y', 'Z', 'energy', 'magnitude', 'type'
    ])
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    
    # 统计每天的事件数量
    daily_counts = df.groupby(['date', 'type']).size().unstack(fill_value=0)
    daily_counts.to_csv(os.path.join(output_dir, 'daily_counts.csv'))
    
    # 可视化部分
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制三个子图
    data_list = [
        (input_seq[-1, 0], "Input"),
        (prediction[-1, 0], "Prediction"),
        (target[-1, 0], "Target")
    ]
    
    for idx, (data, title) in enumerate(data_list, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        nonzero = np.nonzero(data)
        if len(nonzero[0]) > 0:
            scatter = ax.scatter(
                nonzero[0], nonzero[1], nonzero[2],
                c=data[nonzero],
                cmap='viridis',
                marker='o'
            )
            plt.colorbar(scatter, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return {
        'predictions_path': csv_path,
        'daily_counts_path': os.path.join(output_dir, 'daily_counts.csv')
    }