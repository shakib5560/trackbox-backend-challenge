from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from shapely.geometry import Polygon

class ProcessingStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class ObservationStatus(Enum):
    VALID = "VALID"
    INVALID_OR_MISSING = "INVALID_OR_MISSING"

@dataclass
class DetectionResult:
    frame_count: int
    polygon: Polygon
    intersection_area: float
    status: ObservationStatus = ObservationStatus.VALID

@dataclass
class PipelineRunResult:
    status: ProcessingStatus
    valid_results: List[DetectionResult] = field(default_factory=list)
    invalid_count: int = 0
    frames_inspected: int = 0
    error_message: Optional[str] = None
