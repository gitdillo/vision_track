# trackers/csrt.py

import cv2
from .base import TrackingAlgorithmBase


class CSRTTracker(TrackingAlgorithmBase):
    def _create_tracker(self):
        return cv2.legacy.TrackerCSRT_create()
