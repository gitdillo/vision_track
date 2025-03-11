# ML_training/scripts/main.py
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Run ML training pipeline.')
    parser.add_argument('--config', default='config.json', help='Path to config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Call data_manager, network, and trainer functions based on config
    from data_manager import extract_and_split_data
    from network import define_network
    from trainer import train_and_evaluate

    data = extract_and_split_data(config['data_path'])
    network = define_network(config['network_config'])
    train_and_evaluate(network, data, config['training_config'])

if __name__ == '__main__':
    main()
