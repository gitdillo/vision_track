import cv2
import json
import zipfile
import os
import time
import numpy as np
# from datetime import datetime
# from .data_format import DataFormat
# from data_io.data_format import DataFormat
from classic_CV.data_io.data_format import DataFormat

class InputHandler:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.metadata = None
        self.roi_frame = None
        self._current_zip = None
        self.is_live_camera = False  # New flag to track input type

        if isinstance(source, str) and source.endswith(".zip"):
            self._init_from_zip()
            self.is_live_camera = False
        else:
            self._init_from_camera()
            self.is_live_camera = True

    def _init_from_camera(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {self.source}")

    def warm_up(self, duration=2):
        """Warm up camera by discarding initial frames"""
        if not self.is_live_camera:
            return

        print(f"Warming up camera for {duration} seconds...")
        start_time = time.time()
        frames_read = 0

        while time.time() - start_time < duration:
            ret, _ = self.cap.read()
            if ret:
                frames_read += 1
            else:
                break

        print(f"Warmup complete. Discarded {frames_read} frames.")

    def _init_from_zip(self):
        self._current_zip = zipfile.ZipFile(self.source, "r")

        # Load metadata
        with self._current_zip.open(DataFormat.METADATA_JSON) as f:
            self.metadata = json.load(f)

        # Load ROI frame
        with self._current_zip.open(DataFormat.ROI_FRAME) as f:
            self.roi_frame = cv2.imdecode(
                np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR
            )

        # Initialize video readers
        raw_video_data = self._current_zip.read(DataFormat.RAW_VIDEO)
        self.cap = cv2.VideoCapture(
            self._bytes_to_video(raw_video_data), cv2.CAP_FFMPEG
        )

        # Load annotations
        self.annotations = []
        ann_data = self._current_zip.read(DataFormat.ANNOTATIONS_BIN)
        offset = 0
        while offset < len(ann_data):
            # Read header
            header = DataFormat.ANNOTATION_HEADER.unpack_from(ann_data, offset)
            offset += DataFormat.ANNOTATION_HEADER.size
            frame_num, num_ann = header

            # Read annotations
            frame_ann = []
            for _ in range(num_ann):
                ann = DataFormat.ANNOTATION_ITEM.unpack_from(ann_data, offset)
                offset += DataFormat.ANNOTATION_ITEM.size
                frame_ann.append({"bbox": ann[:4], "confidence": ann[4]})
            self.annotations.append(frame_ann)

    def _bytes_to_video(self, data):
        # Helper to convert bytes to temporary video file
        temp_path = f"temp_{DataFormat.RAW_VIDEO}"
        with open(temp_path, "wb") as f:
            f.write(data)
        return temp_path

    def fetch_frame(self):
        if self.cap is None:
            return None, False

        ret, frame = self.cap.read()
        if not ret:
            return None, False

        return frame, True

    def get_annotations(self, frame_number):
        if frame_number < len(self.annotations):
            return self.annotations[frame_number]
        return []

    def get_metadata(self):
        return self.metadata.copy()

    def release(self):
        if self.cap:
            self.cap.release()
        if self._current_zip:
            self._current_zip.close()


class OutputHandler:
    def __init__(self, output_path, fps, frame_size):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.frame_count = 0
        self.roi_frame = None
        self.roi_info = None

        # Temporary files
        self.temp_files = {
            DataFormat.RAW_VIDEO: f"{output_path}_temp_{DataFormat.RAW_VIDEO}",
            DataFormat.ANNOTATED_VIDEO: f"{output_path}_temp_{DataFormat.ANNOTATED_VIDEO}",
            DataFormat.ANNOTATIONS_BIN: f"{output_path}_temp_{DataFormat.ANNOTATIONS_BIN}",
            DataFormat.METADATA_JSON: f"{output_path}_temp_{DataFormat.METADATA_JSON}",
            DataFormat.ROI_FRAME: f"{output_path}_temp_{DataFormat.ROI_FRAME}",
        }

        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*DataFormat.VIDEO_CODEC)
        self.raw_writer = cv2.VideoWriter(
            self.temp_files[DataFormat.RAW_VIDEO], fourcc, fps, frame_size
        )
        self.annotated_writer = cv2.VideoWriter(
            self.temp_files[DataFormat.ANNOTATED_VIDEO], fourcc, fps, frame_size
        )

        # Open annotation file
        self.annotation_file = open(self.temp_files[DataFormat.ANNOTATIONS_BIN], "wb")

    def write_frame(self, frame, annotations):
        # Write raw frame
        self.raw_writer.write(frame)
        
        # Write annotated frame
        annotated_frame = frame.copy()
        for ann in annotations:
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{ann['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        self.annotated_writer.write(annotated_frame)
        
        # Write binary annotations
        header = DataFormat.ANNOTATION_HEADER.pack(self.frame_count, len(annotations))
        self.annotation_file.write(header)
        for ann in annotations:
            data = (*ann['bbox'], ann['confidence'])
            self.annotation_file.write(DataFormat.ANNOTATION_ITEM.pack(*data))
        
        self.frame_count += 1

    def set_roi(self, frame, roi):
        self.roi_frame = frame
        self.roi_info = roi

    def add_file(self, filename, content):
            with open(f"{self.output_path}_temp_{filename}", 'w') as f:
                f.write(content)
            self.temp_files[filename] = f"{self.output_path}_temp_{filename}"

    def finalize(self):
        # Release video writers
        self.raw_writer.release()
        self.annotated_writer.release()
        self.annotation_file.close()

        # Create metadata
        metadata = {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "frame_size": self.frame_size,
            "roi": self.roi_info,
        }
        with open(self.temp_files[DataFormat.METADATA_JSON], "w") as f:
            json.dump(metadata, f)

        # Save ROI frame
        cv2.imwrite(self.temp_files[DataFormat.ROI_FRAME], self.roi_frame)

        # Create ZIP archive
        with zipfile.ZipFile(self.output_path, "w") as zipf:
            # Add all components
            for arcname, temp_path in self.temp_files.items():
                if os.path.exists(temp_path):
                    zipf.write(temp_path, arcname=arcname)

            # Add README
            zipf.writestr(DataFormat.README, DataFormat.get_readme_content())

        # Cleanup temporary files
        for temp_path in self.temp_files.values():
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def __del__(self):
        if hasattr(self, "raw_writer"):
            self.raw_writer.release()
        if hasattr(self, "annotated_writer"):
            self.annotated_writer.release()
        if hasattr(self, "annotation_file"):
            self.annotation_file.close()