# trackers/kcf.py

import cv2
from .base import TrackingAlgorithmBase


class KCFTracker(TrackingAlgorithmBase):
    def _create_tracker(self):
        return cv2.legacy.TrackerKCF_create()
