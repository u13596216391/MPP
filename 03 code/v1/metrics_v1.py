import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
def flatten_tensor(x):
    """将高维张量展平为2维数组"""
    # 保持第一维(batch)不变，将其他维度展平
    return x.reshape(x.shape[0], -1)

# def calculate_metrics(predictions, targets):
#     """计算预测结果的评估指标"""
#     # 转换为numpy数组
#     pred_np = predictions.numpy()
#     target_np = targets.numpy()
    
#     # 展平高维数据为2维数组
#     pred_flat = flatten_tensor(pred_np)
#     target_flat = flatten_tensor(target_np)
    
#     return {
#         'mse': mean_squared_error(target_flat, pred_flat),
#         'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
#         'mae': mean_absolute_error(target_flat, pred_flat),
#         'r2': r2_score(target_flat, pred_flat)
#     }
def calculate_metrics(predictions, targets, threshold=0.8):
    """统一的指标计算函数"""
    # 应用阈值过滤
    filtered_predictions = apply_threshold_filter(predictions, threshold)
    
    # 计算事件数量
    pred_events = torch.sum(filtered_predictions > 0, dim=(2,3,4)).cpu().numpy()
    target_events = torch.sum(targets > 0, dim=(2,3,4)).cpu().numpy()
    
    # 基础指标
    metrics = {
        'mse': mean_squared_error(pred_events.flatten(), target_events.flatten()),
        'mae': mean_absolute_error(pred_events.flatten(), target_events.flatten()),
        'daily_events_ratio': np.mean(pred_events / (target_events + 1e-6))
    }
    
    # IoU指标
    with torch.no_grad():
        intersection = torch.sum(filtered_predictions * targets)
        union = torch.sum(filtered_predictions + targets) - intersection
        iou = (intersection / (union + 1e-6)).item()
        metrics['iou'] = iou
    
    return metrics
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

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    计算评估指标
    Args:
        predictions: 预测张量
        targets: 目标张量
        threshold: 评估阈值
    """
    # 应用阈值过滤
    filtered_predictions = apply_threshold_filter(predictions, threshold)
    
    # 转换为numpy数组
    pred_np = filtered_predictions.numpy()
    target_np = targets.numpy()
    
    # 展平高维数据
    pred_flat = flatten_tensor(pred_np)
    target_flat = flatten_tensor(target_np)
    
    # 计算指标
    metrics = {
        'mse': mean_squared_error(target_flat, pred_flat),
        'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
        'mae': mean_absolute_error(target_flat, pred_flat),
        'r2': r2_score(target_flat, pred_flat)
    }
    
    # 添加IoU相关指标
    with torch.no_grad():
        intersection = torch.sum(filtered_predictions * targets)
        union = torch.sum(filtered_predictions + targets) - intersection
        iou = (intersection / (union + 1e-6)).item()
        metrics['iou'] = iou
    
    return metrics

def calculate_density_penalty(predictions, targets, spatial_dims=(180, 30, 10)):
    """优化的密度惩罚计算"""
    # 计算每个时间步的事件数量
    pred_events = torch.sum(predictions > 0, dim=(2,3,4)).float()  # [batch, channels]
    target_events = torch.sum(targets > 0, dim=(2,3,4)).float()  # [batch, channels]
    
    # 计算比例差异
    ratio = pred_events / (target_events + 1e-6)
    penalty = torch.abs(ratio - 1.0)  # 期望比例为1
    
    return penalty.mean()

def iou_nms_loss(predictions, targets, threshold=0.8):
    """改进的IoU NMS损失函数"""
    # 应用阈值过滤
    filtered_predictions = apply_threshold_filter(predictions, threshold)
    
    # IoU损失
    intersection = torch.sum(filtered_predictions * targets, dim=(2,3,4,5))
    union = torch.sum(filtered_predictions + targets, dim=(2,3,4,5)) - intersection
    iou = intersection / (union + 1e-6)
    iou_loss = 1 - iou.mean()
    
    # 密度惩罚
    density_loss = calculate_density_penalty(filtered_predictions, targets)
    
    return iou_loss, density_loss, filtered_predictions