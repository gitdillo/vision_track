# trackers/mosse.py

import cv2
from .base import TrackingAlgorithmBase

class MOSSETracker(TrackingAlgorithmBase):
    def _create_tracker(self):
        return cv2.legacy.TrackerMOSSE_create()
