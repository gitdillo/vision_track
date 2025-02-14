# File: vision_track/ML_training/scripts/extract_data.py

import os
import sys
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Now we can import from classic_CV
from classic_CV.data_io import InputHandler

def main():
    parser = argparse.ArgumentParser(description='Extract data from annotated datasets.')
    parser.add_argument('-i', '--input', required=True, help='Path to the directory containing dataset zip files.')
    args = parser.parse_args()

    datasets_dir = args.input

    dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('.zip')]
    extracted_data = {}
    
    for dataset_file in dataset_files:
        dataset_path = os.path.join(datasets_dir, dataset_file)
        print(f"Processing dataset: {dataset_file}")
        
        input_handler = InputHandler(dataset_path)
        frame_count = 0
        frames = []
        while True:
            frame, ret = input_handler.fetch_frame()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1

        extracted_data[dataset_file] = {
            'metadata': input_handler.get_metadata(),
            'annotations': input_handler.annotations,
            'frame_count': frame_count
        }
        
        input_handler.release()

    print(f"\nExtracted data from {len(dataset_files)} datasets:")
    for dataset_name, data in extracted_data.items():
        print(f"- {dataset_name}: {len(data['annotations'])} annotations, {data['frame_count']} frames")

if __name__ == '__main__':
    main()
