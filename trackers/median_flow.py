import cv2
from .base import TrackingAlgorithmBase

class MedianFlowTracker(TrackingAlgorithmBase):
    def _create_tracker(self):
        return cv2.legacy.TrackerMedianFlow_create()
