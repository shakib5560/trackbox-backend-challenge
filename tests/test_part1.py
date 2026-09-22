import pytest
import numpy as np
from pydantic import ValidationError
from shapely.geometry import Polygon

from config import AppConfig, FieldDetectorConfig, CropSearchConfig
from detector import FieldDetector
from pipeline import VideoProcessingPipeline
from models import DetectionResult

# 1. Test Configuration (valid)
def test_valid_configuration():
    config = AppConfig(
        video_path="test.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=100),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=10),
        debug_mode=False
    )
    assert config.target_fps == 30

# 2. Test Configuration (invalid rejected)
def test_invalid_configuration_rejected():
    with pytest.raises(ValidationError):
        # min_area cannot be negative
        AppConfig(
            video_path="test.mp4",
            target_fps=30,
            confidence_threshold=0.5,
            field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=-100),
            crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=10),
            debug_mode=False
        )
    
    with pytest.raises(ValidationError):
        # confidence_threshold must be <= 1.0
        AppConfig(
            video_path="test.mp4",
            target_fps=30,
            confidence_threshold=1.5,
            field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=100),
            crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=10),
            debug_mode=False
        )

# 3. Test Detector Abstraction (fake detector)
class FakeDetector:
    def detect(self, frame: np.ndarray):
        # Always return a fixed polygon
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

def test_pipeline_with_fake_detector(mocker):
    config = AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        debug_mode=False
    )
    
    # Mock cv2.VideoCapture to return 2 dummy frames
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    
    # .read() returns (True, frame) twice, then (False, None)
    mock_cap.read.side_effect = [
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (False, None)
    ]
    
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    detector = FakeDetector()
    pipeline = VideoProcessingPipeline(config, detector)
    
    # Mute time.sleep for faster tests
    mocker.patch('time.sleep')
    
    results = pipeline.run()
    
    assert len(results) == 2
    assert results[0].frame_count == 1
    assert results[1].frame_count == 2
    assert isinstance(results[0].polygon, Polygon)
