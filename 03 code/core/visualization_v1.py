import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_prediction(input_seq, prediction, target, save_path=None):
    """
    可视化3D预测结果
    Args:
        input_seq: [seq_len, channels, D, H, W]
        prediction: [seq_len, channels, D, H, W]
        target: [seq_len, channels, D, H, W]
        save_path: 保存路径
    """
    # 选择最后一个时间步和能量通道(channel 0)的数据
    input_data = input_seq[-1, 0]  # [D, H, W]
    pred_data = prediction[-1, 0]
    target_data = target[-1, 0]
    
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制三个子图
    data_list = [
        (input_data, "Input"),
        (pred_data, "Prediction"),
        (target_data, "Target")
    ]
    
    for idx, (data, title) in enumerate(data_list, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        # 获取非零点的坐标和值
        positions = np.where(data > 0)
        if len(positions[0]) > 0:  # 确保有非零点
            x, y, z = positions
            values = data[positions]
            
            # 绘制散点图
            scatter = ax.scatter(
                x, y, z,
                c=values,
                cmap='hot',
                s=50 * values / np.max(values) if np.max(values) > 0 else 50
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
    else:
        plt.show()