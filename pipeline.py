import time
from typing import List
import cv2
from shapely.geometry import Polygon

from config import AppConfig
from models import DetectionResult
from detector import FieldDetector

class VideoProcessingPipeline:
    def __init__(self, config: AppConfig, detector: FieldDetector):
        self.config = config
        self.detector = detector

    def run(self) -> List[DetectionResult]:
        print(f"Starting processing for video: {self.config.video_path}")
        cap = cv2.VideoCapture(self.config.video_path)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return []

        frame_count = 0
        detected_results = []
        
        # Outer boundary based on frame size (1280x720 from prototype)
        # In a real app this might be dynamic or configured
        outer_boundary = Polygon([(0, 0), (1280, 0), (1280, 720), (0, 720)])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            poly = self.detector.detect(frame)

            if poly and poly.is_valid:
                intersection_area = poly.intersection(outer_boundary).area
                result = DetectionResult(
                    frame_count=frame_count,
                    polygon=poly,
                    intersection_area=intersection_area
                )
                detected_results.append(result)

            # Simulate heavy per-frame processing latency
            time.sleep(0.005)

        cap.release()
        print(f"Processed {frame_count} frames. Found {len(detected_results)} boundaries.")
        return detected_results
