# ML_training/scripts/data_manager.py
import os
import zipfile
from vision_track.lib.data_io.data_format import DatasetValidator, DataFormat

def extract_and_split_data(data_path, output_dir, test_ratio=0.2):
    # Validate dataset integrity
    validator = DatasetValidator(data_path)
    if not validator.validate():
        raise ValueError("Dataset is invalid")

    # Extract frames and metadata
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Split data into training and testing sets
    raw_video_path = os.path.join(output_dir, DataFormat.RAW_VIDEO)
    annotated_video_path = os.path.join(output_dir, DataFormat.ANNOTATED_VIDEO)
    annotations_path = os.path.join(output_dir, DataFormat.ANNOTATIONS_BIN)
    metadata_path = os.path.join(output_dir, DataFormat.METADATA_JSON)

    # For simplicity, let's just copy files to separate directories for now
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy files to train and test directories based on test_ratio
    # For now, just copy all files to train directory
    import shutil
    shutil.copy(raw_video_path, train_dir)
    shutil.copy(annotated_video_path, train_dir)
    shutil.copy(annotations_path, train_dir)
    shutil.copy(metadata_path, train_dir)

    return train_dir, test_dir

if __name__ == '__main__':
    data_path = "/vision_track/ML_training/datasets/ON_camMoving_brightBackground.zip"
    output_dir = "/vision_track/ML_training/logs/data_extraction"
    train_dir, test_dir = extract_and_split_data(data_path, output_dir)
    print(f"Training data saved to: {train_dir}")
    print(f"Testing data saved to: {test_dir}")
