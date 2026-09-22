# Automated Pitch Boundary & Crop Engine

A production-oriented video analysis pipeline that detects playing-field boundaries and derives camera crop recommendations from match video feeds.

---

## Architecture

```
main.py              ← thin entry point
  └── VideoProcessingPipeline (pipeline.py)
        ├── FieldDetector protocol (detector.py)
        │     └── SyntheticFieldDetector
        └── Reporter protocol (reporter.py)
              └── HttpMockApiReporter → mock_api
```

**Configuration** is strictly validated at startup via Pydantic (`config.py`). Malformed or missing values fail immediately with a clear error — no silent fallbacks.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Configuration | Pydantic |
| Field detection | OpenCV + Shapely |
| Video I/O | `cv2.VideoCapture` |
| Reporting | HTTP / Flask (`mock_api`) |
| Deployment | Docker Compose |

---

## Key Design Decisions

**Efficiency** — Processing scales with how much video needs inspection, not total video length.
- Frame sampling via `cap.set()` (`inspection_interval_frames`)
- Early termination once enough frames are sampled (`max_inspected_frames`)
- Cheap green-pixel heuristic to skip pitch-less frames before expensive contouring
- Geometry caching for identical consecutive polygons

**Resilience** — Invalid frames (camera cuts, close-ups) are tracked separately and excluded from aggregate metrics. Fatal exceptions surface with full stack traces; nothing is silently swallowed.

**Reporting** — The pipeline reports `STARTED`, periodic `progress`, `SUCCESS`, and `FAILED` events to `mock_api` over the Docker network. A reporting failure never causes a pipeline failure.

---

## Running

### Docker (recommended)

```bash
docker compose up --build
```

Both services start. `mock_api` logs every HTTP request; `runner` logs frame-inspection progress.

### Local (standalone)

```bash
pip install -r requirements.txt
python3 main.py
```

### Tests

```bash
pytest tests/
```

---

## Project Layout

```
starter/
├── main.py                  # Entry point
├── pipeline.py              # Core processing logic
├── detector.py              # FieldDetector protocol + implementation
├── reporter.py              # Reporter protocol + HTTP implementation
├── config.py                # Pydantic configuration models
├── models.py                # Shared result models
├── synthetic_generator.py   # Synthetic video feed generator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── DECISIONS.md             # Design decisions and trade-offs
└── tests/
    ├── test_part1.py
    ├── test_part2.py
    ├── test_part3.py
    └── test_part4.py
```

---

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `MOCK_API_URL` | *(none)* | Set to `http://mock_api:5000` by Docker Compose. If absent, reporting is a no-op. |

---

*See [`DECISIONS.md`](DECISIONS.md) for detailed trade-off documentation.*
