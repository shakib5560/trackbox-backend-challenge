from typing import Protocol, Optional
import time
import cv2
import numpy as np
from shapely.geometry import Polygon
from config import FieldDetectorConfig

class FieldDetector(Protocol):
    def detect(self, frame: np.ndarray) -> Optional[Polygon]:
        ...

class SyntheticFieldDetector:
    def __init__(self, config: FieldDetectorConfig):
        self.config = config

    def detect(self, frame: np.ndarray) -> Optional[Polygon]:
        mask = self._extract_mask(frame)
        
        # Cheap early exit: if less than min_area green pixels, don't do expensive contours
        if cv2.countNonZero(mask) < self.config.min_area:
            return None
            
        # Simulate heavy processing latency for the expensive part of detection
        time.sleep(0.005)
        return self._derive_polygon_from_mask(mask)

    def _extract_mask(self, frame: np.ndarray) -> np.ndarray:
        # Dummy mask generation based on green color thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        return cv2.inRange(hsv, lower_green, upper_green)

    def _derive_polygon_from_mask(self, mask: np.ndarray) -> Optional[Polygon]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > self.config.min_area:
                pts = largest.reshape(-1, 2)
                if len(pts) >= 3:
                    return Polygon(pts)
        return None
