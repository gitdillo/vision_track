# data_format.py
from datetime import datetime
import struct


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
