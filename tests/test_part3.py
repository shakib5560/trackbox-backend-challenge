import pytest
import numpy as np
from shapely.geometry import Polygon

from config import AppConfig, FieldDetectorConfig, CropSearchConfig
from pipeline import VideoProcessingPipeline
from models import ProcessingStatus, ObservationStatus

class FailingDetector:
    def detect(self, frame: np.ndarray):
        raise ValueError("Fatal hardware error simulated")

class FlakyDetector:
    def __init__(self):
        self.call_count = 0

    def detect(self, frame: np.ndarray):
        self.call_count += 1
        # Frame 1: Valid
        # Frame 2: Invalid/Missing
        # Frame 3: Valid
        if self.call_count == 2:
            return None
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

def create_config():
    return AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        inspection_interval_frames=1,
        max_inspected_frames=3
    )

def test_fatal_error_handled_gracefully(mocker):
    config = create_config()
    
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    detector = FailingDetector()
    pipeline = VideoProcessingPipeline(config, detector)
    
    result = pipeline.run()
    
    # Should not crash the test suite, should return FAILED
    assert result.status == ProcessingStatus.FAILED
    assert "Fatal hardware error" in result.error_message
    assert result.frames_inspected == 1

def test_flaky_detection_excludes_invalid_metrics(mocker):
    config = create_config()
    
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (False, None)
    ]
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    detector = FlakyDetector()
    pipeline = VideoProcessingPipeline(config, detector)
    
    result = pipeline.run()
    
    assert result.status == ProcessingStatus.SUCCESS
    assert result.frames_inspected == 3
    assert result.invalid_count == 1
    assert len(result.valid_results) == 2
    
    # Check that frame 1 and 3 are valid, frame 2 was skipped
    assert result.valid_results[0].frame_count == 1
    assert result.valid_results[1].frame_count == 3
