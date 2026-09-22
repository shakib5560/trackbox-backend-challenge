# DECISIONS.md

## Part 2: Performance decisions

### 1. What was identified as expensive operations
- **Sequential frame reading & decoding**: The pipeline read every frame exhaustively. A 10-minute video would always decode 18,000 frames.
- **Constant simulated detector latency**: The `time.sleep(0.005)` simulated a heavyweight GPU or CPU pass, and it was executed unconditionally on every decoded frame, even if the frame was pitch-black.
- **Repeated Geometry Operations**: `poly.intersection(outer_boundary).area` was calculated on every frame, even when the derived boundary was identical to the previous frame's boundary.

### 2. What was optimized
- **Frame Sampling**: Added `inspection_interval_frames` to `AppConfig`. If set to e.g. 5, we use `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` to skip decoding 4 out of every 5 frames entirely.
- **Early Termination**: Added `max_inspected_frames` to `AppConfig`. Once enough frames have been inspected to determine a stable boundary, processing halts entirely (O(1) with respect to total video length).
- **Cheap Early Exit**: Before passing the frame to the expensive `cv2.findContours` (and hitting the `time.sleep` latency), we perform a cheap `cv2.countNonZero(mask)` on the green HSV mask. If it's effectively empty (less than `min_area`), we abort the expensive path instantly.
- **Geometry Caching**: If the detector returns a polygon identical to the previous frame (`poly.equals(last_poly)`), we reuse the previous `intersection_area` instead of performing the expensive Shapely intersection again.

### 3. Why this optimization was chosen
It strikes the balance between skipping unnecessary work and preserving correct output. Using `cap.set` skips decoding overhead. The cheap mask sum acts as a very fast heuristic to avoid computationally expensive operations. Geometry caching leverages the fact that a static camera pointing at a field often produces identical or near-identical masks frame-to-frame.

### 4. Accuracy/Coverage trade-offs
- **Sampling Interval**: Increasing `inspection_interval_frames` means we might miss transient events (e.g., a momentary glitch). However, field detection generally assumes a static or slowly panning camera where high-frequency sampling isn't required.
- **Early Termination**: Halting after `max_inspected_frames` assumes the remainder of the video holds the same scene. If the camera cuts *after* the cap is reached, we wouldn't analyze it. This is a trade-off for scaling independently of video length.

### 5. Assumptions
- It is assumed that `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` is reasonably fast on the target video codec (this is true for standard intra-coded video, but can be slightly slower on certain inter-coded H.264 profiles; however, it's still generally faster than running heavy object detection).
- It is assumed that `poly.equals(last_poly)` is significantly faster than computing `poly.intersection(...)`.

### 6. Validation for production
Before deploying to production, I would benchmark `cap.set` seek times against actual stadium feeds to ensure skipping frames is faster than sequentially reading and dropping them. I would also validate if `max_inspected_frames` provides enough robustness in feeds that include mid-match camera resets.

## Part 3: Failure handling decisions

### 1. Which failures stop the pipeline?
Unexpected runtime errors (e.g., OpenCV memory exhaustion, OS `IOError` when reading the feed) are caught at the top level of `pipeline.run()`, logged as fatal `ERROR`s, and stop the pipeline, returning `ProcessingStatus.FAILED`. This prevents the pipeline from thrashing indefinitely or returning partial results disguised as a full success.

### 2. Which failures are recoverable?
If `detector.detect()` fails to find a polygon because a frame lacks a pitch or contains noise, this is a normal occurrence in the synthetic feed (e.g., camera cuts or close-ups). The pipeline correctly identifies this as an invalid frame, increments an `invalid_count`, and continues processing the next frame.

### 3. Missing/invalid field detection
An invalid observation explicitly skips the `poly.intersection()` math. It is NOT appended to the `valid_results` array. This ensures downstream aggregate metrics are not poisoned by `None` values or hallucinated fallback shapes.

### 4. Zero valid detections
If a video contains zero valid detections, the pipeline completes with `ProcessingStatus.SUCCESS` and an empty `valid_results` list. This correctly reflects reality: the pipeline succeeded in inspecting the video, but the video contained no pitches. 

### 5. Progress exposure
Instead of spamming `print()` for every frame, progress is periodically emitted via `logger.info()` every 100 frames. This gives the operator a heartbeat to track the ratio of valid vs invalid frames over time without polluting the console output.

### 6. Handling unexpected exceptions
In `detector.py`, the extremely broad `except Exception: pass` was removed. If `cv2.findContours` fails unpredictably, it will now crash the frame, bubble up, be logged with full `exc_info` by the pipeline, and safely terminate the run as `FAILED`. No real failure is silently swallowed anymore.
