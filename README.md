⚠️ **STATUS: PROOF OF CONCEPT — DO NOT DEPLOY TO PRODUCTION** ⚠️

# Automated Pitch Boundary & Camera Crop Engine (Prototype)

## Overview

This repository contains the v0.1 prototype for the automated pitch-boundary and camera-crop initiative. It is a
computer-vision pipeline designed to ingest multi-camera match video, detect the playing-field boundary in each
frame, and derive a recommended camera crop layout from that boundary.

Currently, this is a synchronous, single-file script primarily used by the research team to validate the detection
approach before it gets built out into a production pipeline.

## Technology Stack

- **Language:** Python 3.12+
- **Field Detection:** Mock segmentation mask (color-threshold placeholder standing in for a real SAM-style model).
- **Geometry:** Shapely, for polygon derivation and spatial checks.
- **Video I/O:** OpenCV (`cv2.VideoCapture`).

## Features

- **Synthetic Feed Generation:** Generates a dummy match-style video so the script runs standalone with no external
  assets.
- **Field-Boundary Detection:** Extracts a mask per frame and derives a boundary polygon from it.
- **Crop Recommendation Inputs:** The boundary polygons produced here are meant to feed a downstream crop-layout
  step (not yet implemented in this prototype).
- **Execution Metrics:** Reports how many frames were processed and how many boundaries were found.

## Part 1 Architecture and Configuration

The prototype has been refactored into a modular architecture:

```text
Entry Point (main.py)
   ↓
Pipeline (pipeline.py)
   ↓
FieldDetector abstraction (detector.py)
   ↓
SyntheticFieldDetector
```

### Configuration
Configuration is validated using `pydantic` in `config.py`. 
Any invalid configuration (e.g. negative values for minimum area, missing fields) will cause an immediate startup failure with a clear validation error message, preventing the application from continuing with bad state.

Example valid configuration:
```python
AppConfig(
    video_path="synthetic_pitch_feed.mp4",
    target_fps=30,
    confidence_threshold=0.5,
    field_detector=FieldDetectorConfig(type="sam_mask_v1", sport="football", min_area=1000),
    crop_search=CropSearchConfig(aspect_ratio="16:9", padding_px=20),
    debug_mode=True
)
```

## Part 2 Processing Efficiency

To optimize processing and avoid O(N) scaling with respect to video length, several performance optimizations were introduced:

### Previous Behavior
The prototype unconditionally executed the expensive detection logic (`findContours`, heavy CPU simulation latency, and Shapely polygon intersections) on every decoded frame, even if the frame was black or if a steady boundary had already been found. 

### New Behavior
- **Frame Sampling**: By configuring `inspection_interval_frames`, the pipeline uses `cap.set` to skip decoding intermediate frames altogether, reading e.g. only every 10th frame.
- **Cheap Early Exit**: Frames without enough green pixels are discarded before reaching the expensive contour generation and latency path.
- **Geometry Caching**: Identical polygons reuse the previously computed intersection area, saving Shapely evaluation costs.
- **Early Termination**: Setting `max_inspected_frames` halts the video processing entirely once enough frames are sampled, completely avoiding full-video traversal.

### Trade-offs
- Setting a higher `inspection_interval_frames` vastly improves throughput (e.g. 10x faster decoding) but sacrifices frame-perfect boundary adjustments if the camera is actively panning.
- Early termination assumes the rest of the video does not contain a drastically new environment. For the synthetic challenge feed, this easily meets requirements while keeping time complexity bounded.

## Part 3 Failure Handling and Observability

The pipeline is now observable and robust enough for unattended batch processing:

- **Structured Results**: The pipeline returns a `PipelineRunResult` differentiating between `ProcessingStatus.SUCCESS` and `FAILED`.
- **Invalid Observation Segregation**: Frames that fail to produce a valid field boundary (e.g. camera cuts) are explicitly tracked as invalid and are skipped from downstream metric aggregations. They do not crash the pipeline, nor do they silently poison metrics with `0` or `None`.
- **Fatal Error Propagation**: Unexpected exceptions are no longer silently swallowed. They are caught at the pipeline boundary, logged as an `ERROR` with a stack trace, and gracefully terminate the run.
- **Logging vs Printing**: Standard Python `logging` provides periodic `INFO` progress heartbeats (every 100 frames) tracking valid and invalid observation counts, preventing console spam while offering a clear view of processing health.

## Part 4: Reporting to Platform

The pipeline now functions as a microservice running inside a Docker Compose network, communicating its lifecycle and progress to `mock_api` without blocking processing.

- **Docker Architecture**: The `runner` service is built from the `Dockerfile` and talks to the `mock_api` container via the `MOCK_API_URL` environment variable configured in `docker-compose.yml`.
- **API Payloads**: Payload shape is strictly enforced by Pydantic models (`JobEventPayload`, `JobProgressPayload`).
- **Resilience**: If the `mock_api` container crashes, the pipeline issues a warning but continues processing the video successfully, ensuring reporting failures do not corrupt data extraction.

## Running the Code

### With Docker (Full Integration)
To run the entire system including `mock_api` and the pipeline runner:

```bash
docker compose up --build
```
You will see both the `mock_api` logging HTTP requests and the `runner` processing the synthetic video feed.

### Locally (Standalone)
To run the main pipeline locally without the reporter (it will silently mock it):
```bash
python3 main.py
```

To run the unit tests:
```bash
pytest tests/test_part1.py
```

## Installation

Ensure you have a virtual environment set up, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the pipeline with a generated synthetic feed, execute the entry point:

```bash
python synthetic_field_prototype.py
```
