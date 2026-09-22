import pytest
import numpy as np
from shapely.geometry import Polygon

from config import AppConfig, FieldDetectorConfig, CropSearchConfig
from pipeline import VideoProcessingPipeline
from reporter import DummyReporter, JobEventPayload, JobProgressPayload

class SpyingReporter(DummyReporter):
    def __init__(self):
        self.starts = []
        self.progresses = []
        self.successes = []
        self.failures = []

    def report_start(self, job_id: str) -> None:
        self.starts.append(job_id)

    def report_progress(self, job_id: str, inspected: int, valid: int, invalid: int) -> None:
        self.progresses.append((job_id, inspected, valid, invalid))

    def report_success(self, job_id: str, results_summary: dict) -> None:
        self.successes.append((job_id, results_summary))

    def report_failure(self, job_id: str, error_msg: str) -> None:
        self.failures.append((job_id, error_msg))

def test_payload_validation():
    # Validates that payload models require correct types
    payload = JobEventPayload(job_id="test", event_type="STARTED")
    assert payload.job_id == "test"
    
    with pytest.raises(ValueError):
        JobProgressPayload(job_id="1", frames_inspected="not_an_int", valid_detections=0, invalid_detections=0)

def test_pipeline_reports_lifecycle(mocker):
    config = AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        inspection_interval_frames=1,
        max_inspected_frames=101
    )
    
    # Mock video capture to return 101 valid frames
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    class FakeDetector:
        def detect(self, frame):
            return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
            
    reporter = SpyingReporter()
    pipeline = VideoProcessingPipeline(config, FakeDetector(), "job-123", reporter)
    
    pipeline.run()
    
    assert len(reporter.starts) == 1
    assert reporter.starts[0] == "job-123"
    
    # max_inspected_frames = 101, should report progress at 100
    assert len(reporter.progresses) == 1
    assert reporter.progresses[0] == ("job-123", 100, 100, 0)
    
    assert len(reporter.successes) == 1
    assert reporter.successes[0][0] == "job-123"
    assert reporter.successes[0][1]["frames_inspected"] == 101
    assert reporter.successes[0][1]["valid_boundaries"] == 101

def test_pipeline_reports_failure(mocker):
    config = AppConfig(
        video_path="fake.mp4",
        target_fps=30,
        confidence_threshold=0.5,
        field_detector=FieldDetectorConfig(type="fake", sport="soccer", min_area=10),
        crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=0),
        inspection_interval_frames=1,
        max_inspected_frames=10
    )
    
    mock_cap = mocker.MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    
    class FailingDetector:
        def detect(self, frame):
            raise RuntimeError("Fatal Exception")
            
    reporter = SpyingReporter()
    pipeline = VideoProcessingPipeline(config, FailingDetector(), "job-fail", reporter)
    
    pipeline.run()
    
    assert len(reporter.starts) == 1
    assert len(reporter.failures) == 1
    assert reporter.failures[0][0] == "job-fail"
    assert "Fatal Exception" in reporter.failures[0][1]
    assert len(reporter.successes) == 0
