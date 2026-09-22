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
        inspected_count = 0
        detected_results = []
        
        # Outer boundary based on frame size (1280x720 from prototype)
        outer_boundary = Polygon([(0, 0), (1280, 0), (1280, 720), (0, 720)])
        
        # Cache for expensive geometry calculations
        last_poly = None
        last_intersection_area = 0.0

        interval = self.config.inspection_interval_frames
        max_inspected = self.config.max_inspected_frames

        while True:
            # If interval > 1, we can skip decoding intermediate frames entirely
            if interval > 1 and frame_count > 0:
                frame_count += interval - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            inspected_count += 1

            poly = self.detector.detect(frame)

            if poly and poly.is_valid:
                # Reuse expensive intersection calculation if the polygon geometry is identical
                if last_poly and poly.equals(last_poly):
                    intersection_area = last_intersection_area
                else:
                    intersection_area = poly.intersection(outer_boundary).area
                    last_poly = poly
                    last_intersection_area = intersection_area

                result = DetectionResult(
                    frame_count=frame_count,
                    polygon=poly,
                    intersection_area=intersection_area
                )
                detected_results.append(result)
            
            if max_inspected and inspected_count >= max_inspected:
                print(f"Reached max_inspected_frames ({max_inspected}). Stopping early.")
                break

        cap.release()
        print(f"Processed/Skipped to {frame_count} frames. Inspected {inspected_count} frames. Found {len(detected_results)} boundaries.")
        return detected_results
