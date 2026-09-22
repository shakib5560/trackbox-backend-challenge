from typing import Literal, Optional
from pydantic import BaseModel, Field

class FieldDetectorConfig(BaseModel):
    type: str = Field(..., min_length=1)
    sport: str = Field(..., min_length=1)
    min_area: int = Field(..., gt=0)

class CropSearchConfig(BaseModel):
    aspect_ratio: str = Field(..., pattern=r"^\d+:\d+$")
    padding_px: int = Field(..., ge=0)

class AppConfig(BaseModel):
    video_path: str = Field(..., min_length=1)
    target_fps: int = Field(..., gt=0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    field_detector: FieldDetectorConfig
    crop_search: CropSearchConfig
    inspection_interval_frames: int = Field(default=1, ge=1)
    max_inspected_frames: Optional[int] = Field(default=None, ge=1)
    debug_mode: bool = False
