from core.trainer import MicroseismicTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train the microseismic prediction model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()
    
    trainer = MicroseismicTrainer.train_from_config(args.config)
    print("Training completed successfully.")

if __name__ == '__main__':
    main()