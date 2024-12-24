
from core.trainer import MicroseismicTrainer

if __name__ == "__main__":
    cfg_path = "J:\\Microseismic Predict Test\\03 code\\configs\\default.yaml"
    trainer = MicroseismicTrainer.train_from_config(cfg_path)