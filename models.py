from dataclasses import dataclass
from shapely.geometry import Polygon

@dataclass
class DetectionResult:
    frame_count: int
    polygon: Polygon
    intersection_area: float
