# ML_training/scripts/data_manager.py
import os
import zipfile
import cv2
import argparse
import numpy as np
from vision_track.lib.data_io.data_format import DatasetValidator, DataFormat

def extract_and_split_data(data_path, output_dir, test_ratio=0.2, min_sequence_length=100):
    # Validate dataset integrity
    validator = DatasetValidator(data_path)
    if not validator.validate():
        raise ValueError("Dataset is invalid")

    # Extract frames and metadata
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Load raw video and extract frames
    raw_video_path = os.path.join(output_dir, DataFormat.RAW_VIDEO)
    cap = cv2.VideoCapture(raw_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Split frames into sequences of min_sequence_length
    sequences = [frames[i:i+min_sequence_length] for i in range(0, len(frames), min_sequence_length)]

    # Remove sequences that are shorter than min_sequence_length
    sequences = [seq for seq in sequences if len(seq) == min_sequence_length]

    # Split sequences into training and testing sets
    test_sequences = sequences[:int(len(sequences) * test_ratio)]
    train_sequences = sequences[len(test_sequences):]

    # Save frames to train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i, seq in enumerate(train_sequences):
        seq_dir = os.path.join(train_dir, f'seq_{i}')
        os.makedirs(seq_dir, exist_ok=True)
        for j, frame in enumerate(seq):
            cv2.imwrite(os.path.join(seq_dir, f'frame_{j}.png'), frame)

    for i, seq in enumerate(test_sequences):
        seq_dir = os.path.join(test_dir, f'seq_{i}')
        os.makedirs(seq_dir, exist_ok=True)
        for j, frame in enumerate(seq):
            cv2.imwrite(os.path.join(seq_dir, f'frame_{j}.png'), frame)

    return train_dir, test_dir


def main():
    parser = argparse.ArgumentParser(description='Extract and split dataset.')
    parser.add_argument('--data_path', required=True, help='Path to dataset zip file.')
    parser.add_argument('--output_dir', required=True, help='Directory to extract data into.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of data for testing.')
    args = parser.parse_args()

    train_dir, test_dir = extract_and_split_data(args.data_path, args.output_dir, args.test_ratio)
    print(f"Training data saved to: {train_dir}")
    print(f"Testing data saved to: {test_dir}")

if __name__ == '__main__':
    main()