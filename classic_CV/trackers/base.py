# trackers/base.py

import cv2
import numpy as np


class TrackingAlgorithmBase:
    def __init__(self):
        self.trackers = cv2.legacy.MultiTracker_create()
        self.object_ids = []
        self.next_object_id = 0

    def select_ROI(self, frame):
        print("Please select an ROI using your mouse.")
        bbox = cv2.selectROI("Select ROI", frame, False)
        cv2.destroyWindow("Select ROI")
        return bbox

    def initialize(self, frame, bounding_boxes):
        frame = self._ensure_bgr(frame)
        print(f"Frame type: {frame.dtype}, shape: {frame.shape}")

        initialization_successful = False
        for bbox in bounding_boxes:
            print(f"Bounding box: {bbox}, type: {type(bbox)}")

            x, y, w, h = tuple(map(int, bbox))
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > frame.shape[1]
                or y + h > frame.shape[0]
            ):
                print("Invalid bounding box: out of bounds or negative dimensions.")
                continue

            tracker = self._create_tracker()
            try:
                success = self.trackers.add(tracker, frame, (x, y, w, h))
                print(f"Tracker add success: {success}")
                if success:
                    self.object_ids.append(self.next_object_id)
                    self.next_object_id += 1
                    initialization_successful = True
            except Exception as e:
                print(f"Error adding tracker: {e}")

        if not initialization_successful:
            print("No valid trackers were initialized.")

        return initialization_successful

    def update(self, frame):
        frame = self._ensure_bgr(frame)
        success, boxes = self.trackers.update(frame)
        tracked_objects = {}

        if success:
            for i, box in enumerate(boxes):
                x, y, w, h = [int(v) for v in box]
                tracked_objects[self.object_ids[i]] = (x, y, w, h)

        return tracked_objects

    def add_object(self, frame, bounding_box):
        tracker = self._create_tracker()
        success = self.trackers.add(tracker, frame, tuple(bounding_box))
        if success:
            self.object_ids.append(self.next_object_id)
            self.next_object_id += 1
        return success

    def remove_object(self, object_id):
        if object_id in self.object_ids:
            index = self.object_ids.index(object_id)
            del self.object_ids[index]
            print(f"Object ID {object_id} removed. Reinitialization required.")

    def handle_disappearance(self):
        # Placeholder implementation
        pass

    def get_tracked_objects(self):
        return {
            obj_id: bbox
            for obj_id, bbox in zip(self.object_ids, self.trackers.getObjects())
        }

    @staticmethod
    def _ensure_bgr(frame):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _create_tracker(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _create_tracker method")
