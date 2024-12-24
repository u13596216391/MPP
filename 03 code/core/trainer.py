import torch
import yaml
import logging
from pathlib import Path
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.data_processing import MicroseismicDataset, split_and_save_data  # 导入 split_and_save_data
from core.model import SpatioTemporalNet,LSTMCNNModel,UNet3D  # 导入 SpatioTemporalNet, LSTMCNNModel, UNet3D
from core.metrics import calculate_metrics, iou_nms_loss  # 导入 iou_nms_loss
from core.visualization import visualize_prediction
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
import psutil
import numpy as np
from pathlib import Path

class MemoryTracker:
    def __init__(self, logger=None):
        self.logger = logger
        self.gpu_memory_history = []
        self.cpu_memory_history = []
    
    def get_gpu_memory(self):
        """获取GPU显存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            max_memory = torch.cuda.max_memory_allocated() / 1024**2
            return current_memory, max_memory
        return 0, 0
    
    def get_cpu_memory(self):
        """获取CPU内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss / 1024**2  # MB
    
    def log_memory(self, step):
        """记录内存使用"""
        current_gpu, max_gpu = self.get_gpu_memory()
        cpu_memory = self.get_cpu_memory()
        
        self.gpu_memory_history.append((step, current_gpu))
        self.cpu_memory_history.append((step, cpu_memory))
        
        if self.logger:
            self.logger.info(
                f"Step {step} - GPU Memory: {current_gpu:.1f}MB (Max: {max_gpu:.1f}MB), "
                f"CPU Memory: {cpu_memory:.1f}MB"
            )
        
        return current_gpu, cpu_memory
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


class MicroseismicTrainer:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_logging()
        self.setup_device()
        self.setup_paths()
        self.daily_predictions = {} 
        self.memory_tracker = MemoryTracker(self.logger)
        
        # 添加损失权重和阈值
        self.loss_weights = self.cfg.get('loss_weights', {
            'mse': 1.0,
            'iou': 1.0,
            'density': 1.0
        })
        self.threshold = self.cfg.get('prediction_threshold', 0.5)
        
    def load_config(self, cfg_path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/training_{self.cfg['run_name']}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(f"runs/{self.cfg['run_name']}")
    
    def setup_device(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cfg['device'] == 'cuda' 
            else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
    
    def setup_paths(self):
        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self, split_data=False):
        # Split and save data if required
        if split_data:
            split_and_save_data(
                self.cfg['data_path'],  
                output_dir='data/processed/'
            )
        
        # Create datasets and save them as instance attributes
        self.train_dataset = MicroseismicDataset(  # 保存dataset实例
            self.cfg['train_data_path'],
            sequence_length=self.cfg['sequence_length'],
            prediction_days=self.cfg['prediction_days']
        )
        
        self.val_dataset = MicroseismicDataset(    # 保存dataset实例
            self.cfg['val_data_path'],
            sequence_length=self.cfg['sequence_length'],
            prediction_days=self.cfg['prediction_days']
        )
        
        # 打印数据集大小
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg['num_workers']
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers']
        )
    def setup_model(self):
        model_type = self.cfg['model']['type']
        model_params = {
            'spatial_size': tuple(self.cfg['model']['spatial_size']),
            'hidden_dim': self.cfg['model']['hidden_dim'],
            'num_layers': self.cfg['model']['num_layers'],
            'dropout': self.cfg['model']['dropout'],
            'input_seq_len': self.cfg['sequence_length'],
            'output_seq_len': self.cfg['prediction_days']
        }
        
        if model_type == "SpatioTemporalNet":
            self.model = SpatioTemporalNet(**model_params)
        elif model_type == "LSTMCNNModel":
            self.model = LSTMCNNModel(**model_params)
        elif model_type == "UNet3D":
            self.model = UNet3D(
                spatial_size=model_params['spatial_size'],
                input_channels=2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.cfg['learning_rate'],
            weight_decay=self.cfg['weight_decay']
        )
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=3,
            factor=0.5
        )
        
        # 初始化损失函数
        self.mse_criterion = torch.nn.MSELoss()
        self.iou_criterion = iou_nms_loss  # 确保已经导入
        
        # 将模型移到指定设备
        self.model = self.model.to(self.device)
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)
        print(f"Starting epoch {epoch}, total batches: {total_batches}")
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.memory_tracker.log_memory(epoch * len(self.train_loader) + batch_idx)
            print(f"Batch {batch_idx + 1}/{total_batches}")
            print(f"Input shape: {data.shape}, Target shape: {target.shape}")
        
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 调整输出和目标的形状以匹配
            output = output.view(target.shape)
            
            # 计算损失
            mse_loss = self.mse_criterion(output, target)
            iou_loss, density_loss, filtered_output = self.iou_criterion(
                output, target, self.threshold
            )
            
            # 组合损失
            loss = (
                self.loss_weights['mse'] * mse_loss + 
                self.loss_weights['iou'] * iou_loss +
                self.loss_weights['density'] * density_loss
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                self.memory_tracker.clear_memory()
            if batch_idx % self.cfg['log_interval'] == 0:
                self.logger.info(
                f"Batch {batch_idx} - Total Loss: {loss.item():.4f}, "
                f"MSE: {mse_loss.item():.4f}, "
                f"IoU: {iou_loss.item():.4f}, "
                f"Density: {density_loss.item():.4f}"
            )
                if self.writer:
                    current_gpu, cpu_memory = self.memory_tracker.log_memory(
                        epoch * len(self.train_loader) + batch_idx
                    )
                    self.writer.add_scalars(
                        'Memory',
                        {
                            'GPU': current_gpu,
                            'CPU': cpu_memory
                        },
                        epoch * len(self.train_loader) + batch_idx
                    )
                
        return total_loss / len(self.train_loader)

    def save_predictions_to_csv(self, predictions, input_dates, save_path):
        """改进的预测结果保存函数"""
        x_range = (517100, 518100)
        y_range = (4394700, 4395700)
        z_range = (840, 940)
        
        pred_records = []
        
        for batch_idx in range(predictions.shape[0]):
            for seq_idx in range(predictions.shape[1]):
                current_date = (pd.to_datetime(input_dates[batch_idx]) + 
                            pd.Timedelta(days=seq_idx+1)).strftime('%Y/%m/%d')
                
                energy_points = predictions[batch_idx, seq_idx, 0] > 0
                positions = np.where(energy_points)
                
                for idx in range(len(positions[0])):
                    x = x_range[0] + (positions[0][idx] / self.cfg['model']['spatial_size'][0]) * (x_range[1] - x_range[0])
                    y = y_range[0] + (positions[1][idx] / self.cfg['model']['spatial_size'][1]) * (y_range[1] - y_range[0])
                    z = z_range[0] + (positions[2][idx] / self.cfg['model']['spatial_size'][2]) * (z_range[1] - z_range[0])
                    
                    energy = float(predictions[batch_idx, seq_idx, 0][positions[0][idx], positions[1][idx], positions[2][idx]])
                    magnitude = float(predictions[batch_idx, seq_idx, 1][positions[0][idx], positions[1][idx], positions[2][idx]])
                    
                    # 添加额外字段
                    workface = "预测值"
                    tunnel_distance = "未知"
                    time = "12:00:00"
                    
                    pred_records.append({
                        'date': current_date,
                        'time': time,
                        'X': x,
                        'Y': y,
                        'Z': z,
                        'energy': energy,
                        'magnitude': magnitude,
                        'workface': workface,
                        'tunnel_distance': tunnel_distance,
                        'prediction_type': 'model_prediction'
                    })
        
        pred_df = pd.DataFrame(pred_records)
        pred_df.to_csv(save_path, index=False)
        self.logger.info(f"Saved predictions to {save_path}")


    def validate_epoch(self, epoch):
        """修改验证函数以保存预测结果"""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        prediction_visualized = True
        all_predictions = []
        input_dates = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 保存预测结果
                all_predictions.append(output.cpu().numpy())
                
                # 获取输入序列的日期
                batch_dates = [self.val_dataset.data['datetime'].dt.date.unique()[i + self.cfg['sequence_length'] - 1]
                            for i in range(len(data))]
                input_dates.extend(batch_dates)
                # 确保维度匹配
                output = output.view(target.shape)
                
                # 计算损失
                mse_loss = self.mse_criterion(output, target)
                iou_loss, density_loss, filtered_output = self.iou_criterion(output, target, self.threshold)
                
                loss = (
                    self.loss_weights['mse'] * mse_loss + 
                    self.loss_weights['iou'] * iou_loss +
                    self.loss_weights['density'] * density_loss
                )
                
                total_loss += loss.item()
                
                # 计算详细指标
                metrics = calculate_metrics(filtered_output.cpu(), target.cpu())
                all_metrics.append(metrics)
                
                # 只对第一个batch进行可视化
                if not prediction_visualized and epoch % self.cfg['viz_interval'] == 0:
                    save_path = self.save_dir / f'epoch_{epoch}_val_pred.png'
                    try:
                        results = visualize_prediction(
                            data[0].cpu().numpy(),
                            filtered_output[0].cpu().numpy(),
                            target[0].cpu().numpy(),
                            save_path=str(save_path)
                        )
                        
                        if results:
                            self.logger.info(f"Visualization saved to {save_path}")
                            # 记录预测统计信息
                            if 'pred_df' in results:
                                pred_stats = results['pred_df'].agg({
                                    'energy': ['mean', 'std'],
                                    'magnitude': ['mean', 'std']
                                })
                                self.writer.add_scalars('Validation/PredictionStats',
                                                    {'energy_mean': pred_stats['energy']['mean'],
                                                    'magnitude_mean': pred_stats['magnitude']['mean']},
                                                    epoch)
                        
                    except Exception as e:
                        self.logger.error(f"Visualization failed: {str(e)}")
    
                
                # 记录batch级别的指标
                if batch_idx % self.cfg['log_interval'] == 0:
                    self.logger.info(
                        f'Validation Batch: {batch_idx} Loss: {loss.item():.6f} '
                        f'IoU: {metrics["iou"]:.4f}'
                    )
        
        # 计算平均指标
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics]) 
            for k in all_metrics[0].keys()
        }
        
        # 记录到TensorBoard
        for name, value in avg_metrics.items():
            self.writer.add_scalar(f'Validation/{name}', value, epoch)
        
        # 详细日志记录
        self.logger.info(
            f'Validation Epoch: {epoch} '
            f'Avg Loss: {total_loss/len(self.val_loader):.6f} '
            f'Avg IoU: {avg_metrics["iou"]:.4f} '
            f'Avg MSE: {avg_metrics["mse"]:.6f}'
        )
        
        return total_loss / len(self.val_loader), avg_metrics
            

    def visualize_predictions(self, epoch):
        """可视化预测结果"""
        self.model.eval()
        with torch.no_grad():
            # 获取一批验证数据
            val_iter = iter(self.val_loader)
            try:
                data, target = next(val_iter)
            except StopIteration:
                return
            
            # 移动数据到设备并生成预测
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            
            # 调整输出和目标的形状以匹配
            output = output.view(target.shape)
            
            # 转换为 numpy 数组
            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # 创建预测结果可视化
            save_path = self.save_dir / f'prediction_epoch_{epoch}.png'
            results = visualize_prediction(
                data[0].cpu().numpy(),  # 输入序列
                pred_np[0],             # 预测结果
                target_np[0],           # 目标值
                save_path=str(save_path)
            )
            
            if results:
                # 记录预测统计信息
                if 'pred_df' in results:
                    pred_stats = results['pred_df'].agg({
                        'energy': ['mean', 'std'],
                        'magnitude': ['mean', 'std']
                    })
                    self.writer.add_scalars('Validation/PredictionStats',
                                        {'energy_mean': pred_stats['energy']['mean'],
                                        'magnitude_mean': pred_stats['magnitude']['mean']},
                                        epoch)
                
                # 保存图像到 TensorBoard
                if save_path.exists():
                    img = plt.imread(str(save_path))
                    self.writer.add_image('Predictions', img, epoch, dataformats='HWC')
    def train(self):
        self.setup_data(split_data=True)
        self.setup_model()
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=self.cfg.get('early_stopping_patience', 5))
        
        for epoch in range(1, self.cfg['epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch(epoch)  # 修改变量名以匹配返回值
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best.pt', epoch, val_loss, val_metrics)  # 使用val_metrics
            
            # Save checkpoint
            if epoch % self.cfg['save_interval'] == 0:
                self.save_model(
                    f'checkpoint_{epoch}.pt',
                    epoch,
                    val_loss,
                    val_metrics
                )
            
            # Visualize predictions periodically
            if epoch % self.cfg['viz_interval'] == 0:
                self.visualize_predictions(epoch)
            
            # Check early stopping
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Plot losses after training
        self.plot_losses(train_losses, val_losses)
    
    def save_model(self, filename, epoch, val_loss, metrics):
        save_path = self.save_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.cfg
        }, save_path)
        self.logger.info(f'Model saved to {save_path}')
    
    @classmethod
    def train_from_config(cls, cfg_path):
        trainer = cls(cfg_path)
        trainer.train()
        return trainer
    
    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(self.save_dir / 'loss_plot.png')
        plt.close()