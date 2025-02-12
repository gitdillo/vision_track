import cv2
import numpy as np
from .base import TrackingAlgorithmBase

class OpticalFlowTracker(TrackingAlgorithmBase):
    def __init__(self):
        super().__init__()
        self.prev_gray = None
        self.prev_points = None
        self.object_ids = []
        self.next_object_id = 0

    def _create_tracker(self):
        # Not used in Optical Flow Tracker
        pass

    def initialize(self, frame, bounding_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_points = np.array([[(box[0] + box[2]/2, box[1] + box[3]/2)] for box in bounding_boxes], dtype=np.float32)
        self.object_ids = list(range(len(bounding_boxes)))
        self.next_object_id = len(bounding_boxes)
        return True

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or len(self.prev_points) == 0:
            self.prev_gray = gray
            return {}

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)
        
        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]

        tracked_objects = {}
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x, y = new.ravel()
            tracked_objects[self.object_ids[i]] = (int(x - 2), int(y - 2), 4, 4)  # Small bounding box around the point

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.object_ids = [self.object_ids[i] for i in range(len(self.object_ids)) if status[i] == 1]

        return tracked_objects

    def add_object(self, frame, bounding_box):
        x, y, w, h = bounding_box
        new_point = np.array([[x + w/2, y + h/2]], dtype=np.float32)
        self.prev_points = np.vstack((self.prev_points, new_point)) if self.prev_points is not None else new_point
        self.object_ids.append(self.next_object_id)
        self.next_object_id += 1
        return True

    def remove_object(self, object_id):
        if object_id in self.object_ids:
            index = self.object_ids.index(object_id)
            self.object_ids.pop(index)
            self.prev_points = np.delete(self.prev_points, index, axis=0)
