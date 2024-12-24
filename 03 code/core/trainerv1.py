import torch
import yaml
import logging
from pathlib import Path
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.data_processing import MicroseismicDataset, split_and_save_data  # 导入 split_and_save_data
from core.model import SpatioTemporalNet
from core.metrics import calculate_metrics, iou_nms_loss  # 导入 iou_nms_loss
from core.visualization import visualize_prediction
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
                self.cfg['data_path'],  # 使用 self.cfg['data_path']
                output_dir='data/processed/'
            )
        
        # Create datasets
        train_dataset = MicroseismicDataset(
            self.cfg['train_data_path'],
            sequence_length=self.cfg['sequence_length'],
            prediction_days=self.cfg['prediction_days']
        )
        
        val_dataset = MicroseismicDataset(
            self.cfg['val_data_path'],
            sequence_length=self.cfg['sequence_length'],
            prediction_days=self.cfg['prediction_days']
        )
        
        # 打印数据集大小
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg['num_workers']
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers']
        )
    def setup_model(self):
        self.model = SpatioTemporalNet(
            spatial_size=tuple(self.cfg['spatial_size']),
            hidden_dim=self.cfg['hidden_dim'],
            num_layers=self.cfg['num_layers'],
            dropout=self.cfg['dropout'],
            input_seq_len=self.cfg['sequence_length'],
            output_seq_len=self.cfg['prediction_days']
        ).to(self.device)
        
        self.mse_criterion = torch.nn.MSELoss()
        self.threshold = self.cfg.get('prediction_threshold', 0.8)  # 提高默认阈值
        self.loss_weights = {
            'mse': self.cfg.get('loss_weights', {}).get('mse', 0.4),
            'iou': self.cfg.get('loss_weights', {}).get('iou', 0.3),
            'density': self.cfg.get('loss_weights', {}).get('density', 0.3)
        }
        self.iou_criterion = iou_nms_loss
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.cfg['learning_rate'],
            weight_decay=self.cfg['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
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
            
            if batch_idx % self.cfg['log_interval'] == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx}] Loss: {loss.item():.6f}')
                
        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        all_filtered_predictions = []
        all_targets = []
        daily_counts = {'predicted': {}, 'actual': {}}
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 获取过滤后的预测
                _, _, filtered_output = self.iou_criterion(output, target, self.threshold)
                
                # 计算损失
                mse_loss = self.mse_criterion(filtered_output, target)
                iou_loss, density_loss, _ = self.iou_criterion(filtered_output, target, self.threshold)
                
                loss = (
                    self.loss_weights['mse'] * mse_loss + 
                    self.loss_weights['iou'] * iou_loss +
                    self.loss_weights['density'] * density_loss
                )
                
                total_loss += loss.item()
                all_filtered_predictions.append(filtered_output.cpu())
                all_targets.append(target.cpu())
                
                # 统计每天的事件数量
                pred_events = torch.sum(filtered_output > 0, dim=(2,3,4)).cpu().numpy()
                target_events = torch.sum(target > 0, dim=(2,3,4)).cpu().numpy()
                
                for day in range(pred_events.shape[1]):
                    date = f"Day_{epoch}_{day}"
                    daily_counts['predicted'][date] = pred_events[0, day]
                    daily_counts['actual'][date] = target_events[0, day]
        
        # 保存日常统计
        self.daily_predictions[epoch] = daily_counts
        
        # 计算指标
        metrics = calculate_metrics(
            torch.cat(all_filtered_predictions),
            torch.cat(all_targets)
        )
        
        return total_loss / len(self.val_loader), metrics

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
            
            # 转换为 numpy 数组
            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # 创建预测结果可视化
            save_path = self.save_dir / f'prediction_epoch_{epoch}.png'
            results = visualize_prediction(
                data[0].cpu().numpy(),  # 输入序列
                pred_np[0],             # 预测结果
                target_np[0],           # 目标值
                timestamps=None,        # 可选：添加时间戳
                save_path=str(save_path)
            )
            
            # 读取生成的预测结果CSV
            if results and 'predictions_path' in results:
                df = pd.read_csv(results['predictions_path'])
                
                # 确保必要的列存在
                if 'type' in df.columns:
                    pred_mean = df[df['type'] == 'prediction']['energy'].mean()
                    target_mean = df[df['type'] == 'target']['energy'].mean()
                    
                    # 记录到 TensorBoard
                    self.writer.add_scalar('Prediction/Mean_Energy', pred_mean, epoch)
                    self.writer.add_scalar('Target/Mean_Energy', target_mean, epoch)
                    
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
            val_loss, metrics = self.validate_epoch(epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best.pt', epoch, val_loss, metrics)
            
            # Save checkpoint
            if epoch % self.cfg['save_interval'] == 0:
                self.save_model(
                    f'checkpoint_{epoch}.pt',
                    epoch,
                    val_loss,
                    metrics
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