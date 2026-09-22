import sys
import logging
from config import AppConfig
from detector import SyntheticFieldDetector
from pipeline import VideoProcessingPipeline
from synthetic_generator import generate_synthetic_video
from models import ProcessingStatus

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# A fallback dictionary in case we want to emulate environment loading
DEFAULT_CONFIG = {
    "video_path": "synthetic_pitch_feed.mp4",
    "target_fps": 30,
    "confidence_threshold": 0.5,
    "field_detector": {
        "type": "sam_mask_v1",
        "sport": "football",
        "min_area": 1000,
    },
    "crop_search": {
        "aspect_ratio": "16:9",
        "padding_px": 20,
    },
    "debug_mode": True,
}

def load_config() -> AppConfig:
    # In a real application, this would load from os.environ, .env files, etc.
    # Here, we validate the dictionary through Pydantic to fulfill Part 1 requirement
    # "reject malformed values", "fail immediately during application startup"
    return AppConfig(**DEFAULT_CONFIG)

def main():
    # 1. Load and validate configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Startup Error: Invalid configuration.\n{e}")
        sys.exit(1)

    # Helper to generate input file if it doesn't exist locally
    generate_synthetic_video(config.video_path)

    # 2. Initialize dependencies (the single seam)
    detector = SyntheticFieldDetector(config.field_detector)

    # 3. Construct the pipeline
    pipeline = VideoProcessingPipeline(config, detector)

    # 4. Invoke the pipeline
    result = pipeline.run()
    
    if result.status == ProcessingStatus.FAILED:
        logger.error(f"Pipeline failed: {result.error_message}")
        sys.exit(1)
    
    logger.info(f"Pipeline finished with SUCCESS. Valid results: {len(result.valid_results)}, Invalid: {result.invalid_count}")

if __name__ == "__main__":
    main()
