import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

def flatten_tensor(x):
    """将高维张量展平为2维数组"""
    return x.reshape(x.shape[0], -1)

def apply_threshold_filter(predictions, threshold=0.5):
    """
    应用阈值过滤
    Args:
        predictions: 预测张量
        threshold: 过滤阈值
    Returns:
        过滤后的预测张量
    """
    return torch.where(predictions > threshold, predictions, torch.zeros_like(predictions))
def calculate_density_penalty(predictions, targets, spatial_dims=(180, 30, 10)):
    """优化的密度惩罚计算"""
    # 计算每个时间步的事件数量
    pred_events = torch.sum(predictions > 0, dim=(2,3,4)).float()  # [batch, channels]
    target_events = torch.sum(targets > 0, dim=(2,3,4)).float()  # [batch, channels]
    
    # 计算比例差异
    ratio = pred_events / (target_events + 1e-6)
    penalty = torch.abs(ratio - 1.0)  # 期望比例为1
    
    return penalty.mean()

def iou_nms_loss(predictions, targets, threshold=0.1):  # 降低阈值以捕获更多事件
    """改进后的IoU NMS损失函数"""
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
    
    # 数据范围检查
    print(f"Input shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")
    print(f"Value ranges - Predictions: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Value ranges - Targets: [{targets.min():.4f}, {targets.max():.4f}]")
    
    # 应用软阈值
    soft_threshold = lambda x: torch.sigmoid((x - threshold) * 10)
    filtered_predictions = soft_threshold(predictions)
    filtered_targets = soft_threshold(targets)
    
    # 计算IoU (按batch和channel分别计算)
    batch_size, num_channels = predictions.shape[:2]
    total_iou = 0
    count = 0
    
    for b in range(batch_size):
        for c in range(num_channels):
            pred = filtered_predictions[b, c]
            targ = filtered_targets[b, c]
            
            intersection = torch.sum(pred * targ)
            union = torch.sum(pred) + torch.sum(targ) - intersection
            
            if union > 0:
                iou = intersection / union
                total_iou += iou
                count += 1
    
    if count > 0:
        mean_iou = total_iou / count
    else:
        mean_iou = torch.tensor(0.0, device=predictions.device)
    
    print(f"Batch IoU stats - Mean: {mean_iou:.4f}")
    
    # 计算损失
    iou_loss = 1 - mean_iou
    density_loss = calculate_density_penalty(filtered_predictions, filtered_targets)
    
    # 添加平滑项以防止梯度消失
    smoothness_loss = torch.mean(torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1]))
    
    return iou_loss, density_loss, filtered_predictions

def calculate_metrics(predictions, targets, threshold=0.1):
    """计算评估指标，包含准确率、召回率和F1分数"""
    # 应用软阈值
    soft_threshold = lambda x: torch.sigmoid((x - threshold) * 10)
    filtered_predictions = soft_threshold(predictions)
    filtered_targets = soft_threshold(targets)
    
    # 转换为numpy数组
    pred_np = filtered_predictions.detach().cpu().numpy()
    target_np = filtered_targets.detach().cpu().numpy()
    
    # 展平高维数据
    pred_flat = pred_np.reshape(pred_np.shape[0], -1)
    target_flat = target_np.reshape(target_np.shape[0], -1)
    
    # 计算基本指标
    metrics = {
        'mse': mean_squared_error(target_flat, pred_flat),
        'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
        'mae': mean_absolute_error(target_flat, pred_flat),
        'r2': r2_score(target_flat, pred_flat)
    }
    
    # 计算二值化预测和目标
    pred_binary = (pred_flat > 0.5).astype(int)
    target_binary = (target_flat > 0.5).astype(int)
    
    # 计算TP, FP, FN
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))
    
    # 计算精确率、召回率和F1分数
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics.update({
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    })
    
    # 计算IoU
    intersection = np.sum(pred_np * target_np, axis=(2,3,4))
    union = np.sum(pred_np, axis=(2,3,4)) + np.sum(target_np, axis=(2,3,4)) - intersection
    iou = np.mean(intersection / (union + 1e-6))
    metrics['iou'] = float(iou)
    
    return metrics