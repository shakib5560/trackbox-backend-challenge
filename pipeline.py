import logging
from typing import List
import cv2
from shapely.geometry import Polygon

from config import AppConfig
from models import DetectionResult, PipelineRunResult, ProcessingStatus, ObservationStatus
from detector import FieldDetector

logger = logging.getLogger(__name__)

class VideoProcessingPipeline:
    def __init__(self, config: AppConfig, detector: FieldDetector):
        self.config = config
        self.detector = detector

    def run(self) -> PipelineRunResult:
        logger.info(f"Starting processing for video: {self.config.video_path}")
        cap = cv2.VideoCapture(self.config.video_path)

        if not cap.isOpened():
            error_msg = f"Failed to open video stream: {self.config.video_path}"
            logger.error(error_msg)
            return PipelineRunResult(status=ProcessingStatus.FAILED, error_message=error_msg)

        frame_count = 0
        inspected_count = 0
        invalid_count = 0
        detected_results = []
        
        # Outer boundary based on frame size (1280x720 from prototype)
        outer_boundary = Polygon([(0, 0), (1280, 0), (1280, 720), (0, 720)])
        
        # Cache for expensive geometry calculations
        last_poly = None
        last_intersection_area = 0.0

        interval = self.config.inspection_interval_frames
        max_inspected = self.config.max_inspected_frames

        try:
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

                if not (poly and poly.is_valid):
                    # Invalid detection: exclude from metrics, record invalidity, continue
                    invalid_count += 1
                    continue

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
                    intersection_area=intersection_area,
                    status=ObservationStatus.VALID
                )
                detected_results.append(result)
                
                # Periodic progress logging
                if inspected_count % 100 == 0:
                    logger.info(f"Progress: Inspected {inspected_count} frames, {len(detected_results)} valid, {invalid_count} invalid.")

                if max_inspected and inspected_count >= max_inspected:
                    logger.info(f"Reached max_inspected_frames ({max_inspected}). Stopping early.")
                    break

        except Exception as e:
            # Fatal pipeline failure catching any unhandled runtime exceptions
            error_msg = f"Fatal exception during frame processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            cap.release()
            return PipelineRunResult(
                status=ProcessingStatus.FAILED,
                valid_results=detected_results,
                invalid_count=invalid_count,
                frames_inspected=inspected_count,
                error_message=error_msg
            )

        cap.release()
        logger.info(f"Pipeline completed successfully. Inspected {inspected_count} frames. Found {len(detected_results)} valid boundaries, {invalid_count} invalid.")
        
        return PipelineRunResult(
            status=ProcessingStatus.SUCCESS,
            valid_results=detected_results,
            invalid_count=invalid_count,
            frames_inspected=inspected_count
        )
