from core.data_processing import MicroseismicDataset

def main():
    data_path = 'data/processed/train.csv'
    dataset = MicroseismicDataset(data_path)
    print("Coordinates converted and saved successfully.")
    print("Spatial size:", dataset.spatial_size)

if __name__ == '__main__':
    main()