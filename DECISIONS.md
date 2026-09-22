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
