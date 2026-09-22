import pytest
import numpy as np
from pydantic import ValidationError
from shapely.geometry import Polygon

from config import AppConfig, FieldDetectorConfig, CropSearchConfig
from detector import FieldDetector, SyntheticFieldDetector
from pipeline import VideoProcessingPipeline
from models import DetectionResult

def test_efficiency_config_validation():
    # max_inspected_frames must be >= 1
    with pytest.raises(ValidationError):
        AppConfig(
            video_path="test.mp4",
            target_fps=30,
            confidence_threshold=0.5,
            field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=100),
            crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=10),
            inspection_interval_frames=1,
            max_inspected_frames=0
        )
    
    # interval must be >= 1
    with pytest.raises(ValidationError):
        AppConfig(
            video_path="test.mp4",
            target_fps=30,
            confidence_threshold=0.5,
            field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=100),
            crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=10),
            inspection_interval_frames=0
        )

class SpyDetector:
    def __init__(self):
        self.call_count = 0
        
    def detect(self, frame: np.ndarray):
        self.call_count += 1
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

def test_pipeline_interval_skips_processing(mocker):
    config = AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        inspection_interval_frames=3,
        max_inspected_frames=2
    )
    
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    
    # 2 successful reads
    mock_cap.read.side_effect = [
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (False, None)
    ]
    
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    detector = SpyDetector()
    pipeline = VideoProcessingPipeline(config, detector)
    
    results = pipeline.run()
    
    # It reads twice (because max_inspected_frames=2)
    # The detector should be called exactly twice.
    assert detector.call_count == 2
    # Because interval is 3, the frame numbers should be 1 and 4
    assert results[0].frame_count == 1
    assert results[1].frame_count == 4

def test_pipeline_caches_geometry(mocker):
    config = AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        inspection_interval_frames=1,
        max_inspected_frames=3
    )
    
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (True, np.zeros((720, 1280, 3), dtype=np.uint8)),
        (False, None)
    ]
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    detector = SpyDetector()
    pipeline = VideoProcessingPipeline(config, detector)
    
    # We spy on Polygon.intersection
    spy = mocker.spy(Polygon, 'intersection')
    
    results = pipeline.run()
    
    # It processed 3 frames, returning the identical Polygon each time
    assert len(results) == 3
    # intersection should only be called ONCE because of caching
    assert spy.call_count == 1
