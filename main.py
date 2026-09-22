import sys
from config import AppConfig
from detector import SyntheticFieldDetector
from pipeline import VideoProcessingPipeline
from synthetic_generator import generate_synthetic_video

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
        print(f"Startup Error: Invalid configuration.\n{e}")
        sys.exit(1)

    # Helper to generate input file if it doesn't exist locally
    generate_synthetic_video(config.video_path)

    # 2. Initialize dependencies (the single seam)
    detector = SyntheticFieldDetector(config.field_detector)

    # 3. Construct the pipeline
    pipeline = VideoProcessingPipeline(config, detector)

    # 4. Invoke the pipeline
    results = pipeline.run()
    
    print(f"Pipeline finished with {len(results)} results.")

if __name__ == "__main__":
    main()
