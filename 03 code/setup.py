import os
from pathlib import Path

class ProjectSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent

        # 项目目录结构
        self.directories = {
            'data': ['raw', 'processed', 'external'],
            'configs': [],
            'core': [],
            'logs': [],
            'runs': [],
            'outputs': ['models', 'predictions', 'visualizations'],
            'notebooks': [],
            'tests': []
        }

        # 基础文件
        self.base_files = {
            'requirements.txt': self.get_requirements_content(),
            'README.md': self.get_readme_content(),
            '.gitignore': self.get_gitignore_content(),
            'train.py': self.get_train_content(),
            'predict.py': self.get_predict_content(),
            'configs/default.yaml': self.get_config_content()
        }

        # 示例数据文件
        self.example_files = {
            'data/raw/example_data.csv': self.get_example_data_content()
        }

    def create_directories(self):
        """创建项目目录结构"""
        for main_dir, sub_dirs in self.directories.items():
            main_path = self.root_dir / main_dir
            main_path.mkdir(exist_ok=True)
            print(f"Created directory: {main_path}")
            for sub_dir in sub_dirs:
                sub_path = main_path / sub_dir
                sub_path.mkdir(exist_ok=True)
                (sub_path / '.gitkeep').touch()
                print(f"Created sub-directory: {sub_path}")

    def create_base_files(self):
        """创建基础文件"""
        for file_path, content in self.base_files.items():
            full_path = self.root_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            print(f"Created file: {full_path}")

    def create_example_files(self):
        """创建示例数据文件"""
        for file_path, content in self.example_files.items():
            full_path = self.root_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            print(f"Created example file: {full_path}")

    def get_requirements_content(self):
        return """torch>=1.9.0
numpy>=1.19.2
pandas>=1.3.0
matplotlib>=3.4.3
seaborn>=0.11.2
scikit-learn>=0.24.2
pyyaml>=5.4.1
tensorboard>=2.6.0
tqdm>=4.62.3
pytest>=6.2.5"""

    def get_readme_content(self):
        return """# 微震预测项目\n\n此项目用于预测微震事件，包含数据处理、模型训练、预测和可视化模块。"""

    def get_gitignore_content(self):
        return """# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
ENV/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Outputs
outputs/models/*
outputs/predictions/*
outputs/visualizations/*
!outputs/models/.gitkeep
!outputs/predictions/.gitkeep
!outputs/visualizations/.gitkeep

# Logs
logs/*
!logs/.gitkeep
runs/*
!runs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db"""

    def get_config_content(self):
        return """# 配置文件示例
run_name: microseismic_prediction_v1
device: cuda
train_data_path: data/processed/train.csv
val_data_path: data/processed/val.csv
batch_size: 32
epochs: 100
learning_rate: 0.001"""

    def get_train_content(self):
        return """from core.trainer import MicroseismicTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train the microseismic prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    trainer = MicroseismicTrainer.train_from_config(args.config)
    print("Training completed successfully.")

if __name__ == '__main__':
    main()"""

    def get_predict_content(self):
        return """import argparse
import torch
from core.model import SpatioTemporalNet
from core.data_processing import load_test_data

def main():
    parser = argparse.ArgumentParser(description='Predict using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to the test data CSV file')
    parser.add_argument('--output', type=str, default='outputs/predictions', help='Output directory for predictions')
    args = parser.parse_args()

    print("Prediction logic to be implemented.")

if __name__ == '__main__':
    main()"""

    def get_example_data_content(self):
        return """date,time,X,Y,Z,energy,magnitude,working_face,face_distance,tunnel_distance
2023/7/4,0:37:26,517689,4395195,872,654.21,0.5,6207,247.6,18
2023/7/4,1:15:43,517690,4395198,875,732.45,0.6,6207,245.8,17.5"""

    def setup(self):
        """执行完整的项目设置"""
        print("Starting project setup...")
        self.create_directories()
        self.create_base_files()
        self.create_example_files()
        print("\nProject setup completed successfully!")

def main():
    setup = ProjectSetup()
    setup.setup()

if __name__ == "__main__":
    main()
