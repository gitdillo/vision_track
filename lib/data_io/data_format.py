# File: vision_track/lib/data_io/data_format.py
from datetime import datetime
import struct
import zipfile
import json


class DataFormat:
    """Centralized data format specification for input/output handling"""

    # File names and paths
    TIMESTAMP_FORMAT = "%Y%m%d-%H_%M_%S"
    RAW_VIDEO = "raw_video.avi"
    ANNOTATED_VIDEO = "annotated_video.avi"
    ANNOTATIONS_BIN = "annotations.bin"
    METADATA_JSON = "metadata.json"
    ROI_FRAME = "roi_frame.png"
    README = "README.txt"

    # Metadata structure
    METADATA_KEYS = ["frame_count", "fps", "frame_size", "roi"]

    # Binary annotation format using struct module
    ANNOTATION_HEADER = struct.Struct("<IH")  # Frame number (I), num_annotations (H)
    ANNOTATION_ITEM = struct.Struct("<4f f")  # bbox (4f), confidence (f)

    # Video codec settings
    VIDEO_CODEC = "HFYU"  # HuffYUV lossless codec
    VIDEO_CODEC_EXTENSION = ".avi"

    @classmethod
    def get_readme_content(cls):
        return f"""This archive contains tracking data in the following structure:
1. {cls.RAW_VIDEO}: Raw video footage ({cls.VIDEO_CODEC} codec)
2. {cls.ANNOTATED_VIDEO}: Annotated video with tracking overlay
3. {cls.ANNOTATIONS_BIN}: Binary tracking data with format:
   - Header: {cls.ANNOTATION_HEADER.format} (frame_number, num_annotations)
   - Per annotation: {cls.ANNOTATION_ITEM.format} (x, y, w, h, confidence)
4. {cls.METADATA_JSON}: JSON metadata with tracking parameters
5. {cls.ROI_FRAME}: Initial ROI selection frame

All numeric values are little-endian. Video frame size and FPS are stored in metadata.
"""


class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def validate_zip_structure(self):
        required_files = [
            DataFormat.RAW_VIDEO,
            DataFormat.ANNOTATED_VIDEO,
            DataFormat.ANNOTATIONS_BIN,
            DataFormat.METADATA_JSON,
            DataFormat.ROI_FRAME,
            DataFormat.README
        ]
        
        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            return all(file in zip_ref.namelist() for file in required_files)

    def validate_metadata(self):
        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            with zip_ref.open(DataFormat.METADATA_JSON) as f:
                metadata = json.load(f)
                return set(metadata.keys()) == set(DataFormat.METADATA_KEYS)

    def validate_annotations(self):
        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            with zip_ref.open(DataFormat.ANNOTATIONS_BIN) as f:
                annotations_data = f.read()  # Read the entire file
                offset = 0
                while offset < len(annotations_data):
                    # Read header
                    if len(annotations_data) - offset < DataFormat.ANNOTATION_HEADER.size:
                        break  # Not enough data left for a full header
                    header = DataFormat.ANNOTATION_HEADER.unpack_from(annotations_data, offset)
                    offset += DataFormat.ANNOTATION_HEADER.size
                    num_annotations = header[1]
                    for _ in range(num_annotations):
                        if len(annotations_data) - offset < DataFormat.ANNOTATION_ITEM.size:
                            break  # Not enough data left for an annotation item
                        annotation = DataFormat.ANNOTATION_ITEM.unpack_from(annotations_data, offset)
                        offset += DataFormat.ANNOTATION_ITEM.size
                return True  # Add more checks as needed
    
    def validate_annotations(self):
        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            with zip_ref.open(DataFormat.ANNOTATIONS_BIN) as f:
                annotations_data = f.read()  # Read the entire file
                offset = 0
                while offset < len(annotations_data):
                    # Read header
                    if len(annotations_data) - offset < DataFormat.ANNOTATION_HEADER.size:
                        break  # Not enough data left for a full header
                    header = DataFormat.ANNOTATION_HEADER.unpack_from(annotations_data, offset)
                    offset += DataFormat.ANNOTATION_HEADER.size
                    num_annotations = header[1]
                    for _ in range(num_annotations):
                        if len(annotations_data) - offset < DataFormat.ANNOTATION_ITEM.size:
                            break  # Not enough data left for an annotation item
                        annotation = DataFormat.ANNOTATION_ITEM.unpack_from(annotations_data, offset)
                        offset += DataFormat.ANNOTATION_ITEM.size
                return True  # Add more checks as needed


    def validate(self):
        return self.validate_zip_structure() and self.validate_metadata() and self.validate_annotations()