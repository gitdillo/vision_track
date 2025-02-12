import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from .base import TrackingAlgorithmBase


class CentroidTracker(TrackingAlgorithmBase):
    def __init__(self, max_disappeared=50):
        super().__init__()
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = max_disappeared

    def _create_tracker(self):
        # Not used in Centroid Tracker
        pass

    def initialize(self, frame, bounding_boxes):
        for bbox in bounding_boxes:
            self.add_object(frame, bbox)
        return True

    def add_object(self, frame, bounding_box):
        x, y, w, h = bounding_box
        centroid = (int(x + w / 2), int(y + h / 2))
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def update(self, frame):
        # This method should be called with detected objects
        # For simplicity, we'll just return the current objects
        return {obj_id: (x - 2, y - 2, 4, 4) for obj_id, (x, y) in self.objects.items()}

    def update_centroids(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.maxDisappeared:
                    self.remove_object(object_id)
            return self.objects

        input_centroids = np.zeros((len(centroids), 2), dtype="int")
        for i, (x, y) in enumerate(centroids):
            input_centroids[i] = (x, y)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.add_object(None, (*input_centroids[i], 0, 0))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.maxDisappeared:
                        self.remove_object(object_id)
            else:
                for col in unused_cols:
                    self.add_object(None, (*input_centroids[col], 0, 0))

        return self.objects
