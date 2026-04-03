# Code Structure & Architecture - MANUS

Comprehensive developer guide to MANUS architecture, module organization, and extension points.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Breakdown](#module-breakdown)
3. [Data Flow](#data-flow)
4. [Extension Guide](#extension-guide)
5. [Testing Guide](#testing-guide)
6. [Performance Optimization](#performance-optimization)
7. [Contributing](#contributing)

---

## Architecture Overview

MANUS follows a **modular, pipeline-based architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                   main.py (Orchestrator)            │
│  - Coordinates all modules                          │
│  - Manages main event loop                          │
│  - Handles I/O (camera, display)                    │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┬─────────────────┬──────────────────┐
       │                │                 │                  │
       ▼                ▼                 ▼                  ▼
   ┌────────┐    ┌──────────────┐  ┌───────────┐    ┌────────────────┐
   │ Camera │    │ MediaPipe    │  │ Gesture   │    │ Action         │
   │ Input  │───▶│ Hand Track   │─▶│ Detection │───▶│ Executor       │
   └────────┘    └──────────────┘  └──────┬────┘    └────────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Validation   │
                                    │ & Filtering  │
                                    └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Kalman       │
                                    │ Smoothing    │
                                    └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ HUD Render   │
                                    │ & Display    │
                                    └──────────────┘
```

### Core Design Principles

1. **Modularity**: Each module has single responsibility
2. **Decoupling**: Modules communicate through clean interfaces
3. **Testability**: All components can be tested independently
4. **Configuration**: All tuning in `config.py`, no magic numbers in code
5. **Logging**: All significant events logged for debugging
6. **Performance**: Optimized critical path (detection → action)

---

## Module Breakdown

### 1. `config.py` - Configuration

**Purpose**: Centralized configuration management for production tuning.

**Key Variables**:
- Camera settings (resolution, FPS, device index)
- MediaPipe settings (complexity, confidence thresholds)
- Kalman filter tuning (noise matrices)
- Gesture thresholds (based on hand scale)
- Cooldowns and timing constraints
- UI settings and colors
- Logging configuration

**How to Use**:
```python
import config

# Read settings
print(config.CURSOR_SMOOTHING)  # 0.35

# Update runtime settings
config.runtime_config.smoothing = 0.5

# Access nested configs
print(config.TARGET_FPS)
```

**Extension Points**:
- Add new gesture thresholds
- Add new action cooldowns
- Create preset profiles (gaming, accessibility, office)

### 2. `logging/logger.py` - Structured Logging

**Purpose**: Consistent, production-grade logging with file rotation.

**Features**:
- Colored console output
- File logging with rotation
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Singleton pattern (one logger instance)
- Performance metrics logging
- Structured error messages with context

**Usage**:
```python
from logging.logger import get_logger

logger = get_logger()
logger.info("Application started")
logger.debug("Frame processing: 32ms")
logger.error("Camera disconnected", extra_data)
logger.log_gesture("pointing", confidence=0.95)
logger.log_performance(fps=30, latency_ms=45, hand_detected=True)
```

**Files**:
- `data/hand_tracking_log.txt` - Runtime logs (rotated)
- Check logs with: `tail -f data/hand_tracking_log.txt`

### 3. `filters/kalman_filter.py` - Cursor Smoothing

**Purpose**: Smooth, stable cursor tracking with prediction.

**Key Features**:
- Constant velocity motion model
- Adaptive noise based on hand velocity
- Position bounds checking
- Velocity estimation
- Settable process/measurement noise

**Architecture**:
```
Measurement (noisy hand position)
    ↓
Prediction (motion model)
    ↓
Innovation (difference between predicted and measured)
    ↓
Correction (Kalman gain weighting)
    ↓
Smoothed position with velocity estimate
```

**Usage**:
```python
from filters.kalman_filter import KalmanFilter

kf = KalmanFilter(
    process_noise=1e-4,      # Trust motion model more
    measurement_noise=1.0,   # Trust measurements
    adaptive_noise=True      # Adapt based on velocity
)

# Update with new measurement
x, y = kf.update([100, 200])

# Get current state
pos = kf.get_position()  # (x, y)
vel = kf.get_velocity()  # (vx, vy)
uncertainty = kf.get_estimated_covariance()  # (var_x, var_y)

# Bounds clamping
x, y = kf.update([100, 200], ensure_bounds=(0, 1920, 0, 1080))
```

**Tuning**:
```python
# For jittery cursor: increase measurement noise
KF_MEASUREMENT_NOISE = 2.0  # Trust fewer measurements

# For laggy cursor: decrease process noise
KF_PROCESS_NOISE = 1e-5  # Trust motion model more
```

### 4. `gesture/gesture_detector.py` - Gesture Recognition

**Purpose**: Detect hand gestures from landmark positions.

**Supported Gestures**:
- Fist (all fingers down)
- Pointing (index up, others down)
- Peace (index+middle up)
- Right-click (index+middle up, ring+pinky down)
- Enter (index+middle+ring up, pinky down, thumb folded)
- Thumbs up
- Shaka (thumb+pinky up)
- Rock (index+pinky up)
- Three-finger click

**How It Works**:
1. Calculate hand base scale (wrist to knuckles distance)
2. For each gesture: check finger positions relative to scale
3. Return confidence scores (0.0-1.0) for all gestures
4. Best matching gesture selected

**Usage**:
```python
from gesture.gesture_detector import GestureDetector

detector = GestureDetector()

# Detect all gestures
scores = detector.detect(landmarks)
# Returns: {'fist': 0.95, 'pointing': 0.2, 'peace': 0.1, ...}

# Detect scroll gesture specifically
direction, confidence = detector.detect_scroll(landmarks)
# Returns: ('up', 0.8) or ('down', 0.7) or (None, 0.0)

# Get palm position in screen coordinates
palm_x, palm_y = detector.get_hand_palm_position(landmarks, 1920, 1080)
```

**Gesture Thresholds**:
All thresholds in `config.py` scale with hand size for robustness:

```python
GESTURE_FINGER_UP_THRESHOLD = 0.2  # Adaptive Y-distance
base_scale = get_hand_base_scale(landmarks)
actual_threshold = GESTURE_FINGER_UP_THRESHOLD * base_scale
```

**Adding New Gesture**:
```python
def _detect_my_gesture(self, landmarks, base_scale):
    """My custom gesture detector."""
    thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
    
    checks = [
        finger_is_up(landmarks, 8, 6, thr),     # Index up
        finger_is_down(landmarks, 12, 10, thr),  # Middle down
        # ... more checks
    ]
    return get_confidence_score(landmarks, checks)
```

### 5. `gesture/gesture_validator.py` - Validation & Cooldowns

**Purpose**: Reduce false positives through multi-frame validation and cooldown management.

**GestureValidator**:
- Requires gesture to be detected N consecutive frames
- Calculates average confidence over window
- Only validates if above confidence threshold
- Tracks gesture transitions (edge-triggered)

**CooldownTracker**:
- Per-gesture cooldown tracking
- Prevents rapid re-triggering
- Stores per-action cooldown times

**Usage**:
```python
from gesture.gesture_validator import GestureValidator, CooldownTracker

validator = GestureValidator(confirmation_frames=3)
cooldowns = CooldownTracker()

# Check if gesture valid over multiple frames
confirmed_gesture, confidence = validator.update(
    current_gesture='pointing',
    confidence=0.95
)
# Returns: ('pointing', 0.93) if confirmed, (None, 0.0) if not

# Check if action ready (not on cooldown)
import time
now = time.time()

if cooldowns.is_ready('click', now):
    # Execute click
    cooldowns.trigger('click', now)

# Get remaining cooldown
remaining = cooldowns.get_remaining_cooldown('click', now)
```

### 6. `actions/action_handler.py` - Command Execution

**Purpose**: Execute mouse and keyboard commands safely with logging.

**Features**:
- Bounds checking for mouse position
- Thread-safe browser open
- Action counting and logging
- Error handling and recovery

**API**:
```python
from actions.action_handler import ActionHandler

handler = ActionHandler()

# Mouse movement
handler.move_mouse(100, 200)

# Clicks
handler.click_left()
handler.click_right()
handler.click_middle()

# Drag
handler.mouse_down()
# ... move mouse
handler.mouse_up()

# Scrolling
handler.scroll('up', amount=3)
handler.scroll('down', amount=5)

# Keyboard
handler.press_key('enter')
handler.press_key('pageup')
handler.press_key('escape')
handler.type_text("Hello World")

# Browser
handler.open_browser("https://example.com")

# Diagnostics
handler.get_action_count()  # Total actions executed
handler.get_screen_size()    # (width, height)
handler.get_mouse_position() # (x, y)
```

### 7. `ui/hud_renderer.py` - HUD Display

**Purpose**: Render information overlay on camera feed.

**Features**:
- Gesture and FPS display
- Hand detection indicator
- Hand skeleton rendering
- Debug info overlay
- Status indicators (active/paused)

**Usage**:
```python
from ui.hud_renderer import HUDRenderer

hud = HUDRenderer()

# Render full HUD
frame = hud.render(
    frame,
    gesture_status='pointing',
    fps=30.5,
    hand_detected=True,
    paused=False,
    extra_info={'Confidence': '0.95', 'Position': '(100, 200)'}
)

# Draw hand skeleton
connections = HUDRenderer.get_hand_connections()
frame = hud.render_hand_landmarks(frame, landmarks, connections)

# Draw palm center
frame = hud.render_palm_center(frame, palm_x, palm_y, radius=5)

cv2.imshow("MANUS", frame)
```

### 8. `ui/settings_gui.py` - Settings Window

**Purpose**: Runtime configuration GUI (tkinter-based).

**Features**:
- Sliders for continuous parameters (smoothing, friction, thresholds)
- Buttons for discrete options (camera resolution)
- Status indicator (active/paused)
- Runs in separate thread (responsive)

**Usage**:
```python
from ui.settings_gui import SettingsGUI
import config

gui = SettingsGUI(config.runtime_config)
gui.start()  # Open window
# ... user adjusts settings
gui.stop()   # Close window
```

**Threading Model**:
- GUI runs on separate thread
- Settings update main thread's `runtime_config` object
- Main thread reads updated values each frame
- Thread-safe via Python's GIL

### 9. `main.py` - Main Orchestrator

**Purpose**: Coordinate all modules, manage event loop, handle I/O.

**Key Responsibilities**:
1. Initialize camera, MediaPipe, all modules
2. Main event loop (capture → detect → filter → execute → render)
3. Handle keyboard input (ESC to exit)
4. Performance monitoring (FPS, latency)
5. Error handling and cleanup

**Main Loop Flow**:
```python
while True:
    # 1. Capture
    ret, frame = cap.read()
    
    # 2. Detect (MediaPipe)
    results = hands.process(rgb_frame)
    
    # 3. Process gestures
    if hand_detected:
        gestures = detector.detect(landmarks)
        validated = validator.update(gestures, confidence)
    
    # 4. Execute actions
    if validated_gesture:
        action_handler.execute(gesture)
    
    # 5. Render
    frame = hud_renderer.render(frame, ...)
    cv2.imshow(frame)
    
    # 6. Check exit
    if cv2.waitKey(1) == 27:  # ESC
        break
```

---

## Data Flow

### Gesture Recognition Pipeline

```
Camera Frame (RGB, 640×480)
    ↓
MediaPipe Hand Detection
    ├─ 21 landmarks per hand
    ├─ x, y, z coordinates (normalized 0-1)
    └─ confidence scores
    ↓
Hand Scale Calculation
    └─ Distance from wrist to MCPs
    ↓
Gesture Detection
    ├─ Calculate all gesture confidence scores
    └─ Get scroll direction (if applicable)
    ↓
Gesture Validation
    ├─ Check N-frame consistency
    ├─ Apply confidence threshold
    └─ Return confirmed gesture or None
    ↓
Action Execution
    ├─ Check cooldown tracker
    ├─ Execute appropriate action
    └─ Update cooldown timer
    ↓
Kalman Filtering (for cursor)
    ├─ Input: Palm position
    ├─ Output: Smoothed cursor position
    └─ Estimate velocity
    ↓
Mouse Movement
    └─ Move to filtered position
    ↓
HUD Rendering
    ├─ Draw gesture status
    ├─ Draw hand landmarks
    └─ Display FPS/diagnostics
    ↓
Display Output
```

### State Management

```
Hand Tracking State:
├─ is_detected: bool
├─ landmarks: List[Landmark]
└─ updated_time: float

Gesture State:
├─ current: str
├─ confidence: float
├─ frame_count: int
└─ last_confirmed_time: float

Cursor State:
├─ position: (x, y)
├─ velocity: (vx, vy)
├─ clicking: bool
└─ kalman_state: KalmanState

Gesture Control State:
├─ active: bool (toggle with fist)
├─ fist_hold_start: float
└─ prev_enter_state: bool
```

---

## Extension Guide

### Adding New Gestures

**Step 1**: Define gesture in `gesture_detector.py`:

```python
def _detect_custom_gesture(self, landmarks, base_scale):
    """Custom gesture with specific finger positions."""
    thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
    
    # Your gesture logic
    checks = [
        finger_is_up(landmarks, 8, 6, thr),      # Check 1
        finger_is_down(landmarks, 12, 10, thr),  # Check 2
        # ... more checks
    ]
    return get_confidence_score(landmarks, checks)
```

**Step 2**: Add to detection in `detect()` method:

```python
def detect(self, landmarks):
    results = {
        # ... existing gestures
        'custom': self._detect_custom_gesture(landmarks, base_scale),
    }
    return results
```

**Step 3**: Add action in `main.py`:

```python
elif validated_gesture == 'custom':
    if cooldown_tracker.is_ready('custom_action', current_time):
        # Execute custom action
        action_handler.custom_method()
        cooldown_tracker.trigger('custom_action', current_time)
```

**Step 4**: Add cooldown in `config.py`:

```python
COOLDOWN_CUSTOM_ACTION = 1.0
```

### Adding New Actions

`actions/action_handler.py`:

```python
def take_screenshot(self):
    """Custom action: take screenshot."""
    try:
        import subprocess
        if sys.platform == 'win32':
            subprocess.run(['screenshot.exe'])
        else:
            subprocess.run(['gnome-screenshot'])
        self.action_count += 1
        logger.log_action('screenshot')
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
```

Use in `main.py`:

```python
elif gesture == 'rock' and config.ENABLE_SPECIAL_GESTURES:
    action_handler.take_screenshot()
```

### Creating Gesture Profiles

`config.py`:

```python
class GestureProfile:
    """Tuning preset for different use cases."""
    
    GAMING = {
        'cursor_smoothing': 0.2,  # Snappy for gaming
        'gesture_thresholds': 0.15,  # Fast gesture detection
        'confirmation_frames': 1,  # Instant response
    }
    
    ACCESSIBILITY = {
        'cursor_smoothing': 0.7,  # Very smooth
        'gesture_thresholds': 0.3,  # Forgiving
        'confirmation_frames': 5,  # Stable
    }
    
    OFFICE = {
        'cursor_smoothing': 0.5,  # Balanced
        'gesture_thresholds': 0.2,
        'confirmation_frames': 3,
    }

def apply_profile(profile_name):
    """Apply tuning preset."""
    profile = getattr(GestureProfile, profile_name.upper())
    for key, val in profile.items():
        if hasattr(config.runtime_config, key):
            setattr(config.runtime_config, key, val)
```

---

## Testing Guide

### Unit Tests

Structure: `tests/test_*.py`

**Example - Gesture Detector Tests**:

```python
class TestGestureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GestureDetector()
        self.landmarks = create_test_landmarks()
    
    def test_fist_detection(self):
        fist_lm = create_fist_pose()
        scores = self.detector.detect(fist_lm)
        
        self.assertGreater(scores['fist'], 0.7)
        self.assertLess(scores['pointing'], 0.3)
```

**Run tests**:
```bash
python -m pytest tests/ -v
python -m pytest tests/test_kalman_filter.py::TestKalmanFilter::test_convergence -v
```

### Integration Tests

Test gesture → action mapping:

```python
def test_fist_click_sequence():
    # Simulate 5 frames of fist gesture
    frames = [create_fist_pose() for _ in range(5)]
    
    for frame in frames:
        gesture = detector.detect(frame)
        validated = validator.update(gesture, confidence=0.9)
    
    # Should detect fist click
    assert validated == 'fist'
```

### Performance Testing

```python
import timeit

# Measure gesture detection speed
def test_gesture_detection_speed():
    landmarks = create_realistic_landmarks()
    
    time_ms = timeit.timeit(
        lambda: detector.detect(landmarks),
        number=1000
    ) / 1000
    
    assert time_ms < 10, f"Too slow: {time_ms}ms"  # <10ms target
```

### Mock Testing

```python
from unittest.mock import Mock

def test_action_handler_with_mock():
    handler = ActionHandler()
    
    with patch('pyautogui.moveTo') as mock_move:
        handler.move_mouse(100, 200)
        mock_move.assert_called_once_with(100, 200, duration=0)
```

---

## Performance Optimization

### Profiling

```bash
# Install profiler
pip install py-spy

# Profile running application
py-spy record -o profile.svg python main.py

# Check which functions take most time
# Open profile.svg in browser
```

### Optimization Strategies

**1. MediaPipe Optimization**:
```python
# Use lite model (0) instead of full (1)
MP_MODEL_COMPLEXITY = 0  # ~5x faster, slightly less accurate

# Skip frames (process every 2nd frame)
if frame_count % 2 == 0:
    results = hands.process(rgb)
else:
    # Reuse previous results
    pass
```

**2. Resolution Optimization**:
```python
# Lower resolution = faster (quadratic)
# 1920×1080 → 4x slowdown vs 960×540
CAMERA_RESOLUTION_WIDTH = 640   # Good balance
CAMERA_RESOLUTION_HEIGHT = 480
```

**3. Gesture Detection Optimization**:
```python
# Cache base scale calculation
base_scale = get_hand_base_scale(landmarks)  # ~1-2ms
# Don't recalculate in each gesture check

# Use NumPy for batch calculations
thresholds = np.array([...])  # Vectorized operations
```

**4. Kalman Filter Optimization**:
```python
# Use lower precision if acceptable
self.x = self.x.astype(np.float32)  # vs float64
self.P = self.P.astype(np.float32)

# Avoid matrix inversion when possible
# Use LU decomposition or Cholesky for speed
```

### Bottleneck Analysis

Typical frame breakdown (640×480 @ 30 FPS):
- Camera capture: 5-10ms
- MediaPipe detection: 15-25ms (main bottleneck)
- Gesture recognition: 2-3ms
- Kalman filtering: <1ms
- UI rendering: 5-10ms
- **Total**: ~30-50ms per frame

To improve:
1. MediaPipe is the main bottleneck → reduce complexity/resolution
2. Gesture detection is fast → can add complexity safely
3. Kalman filter is fast → use higher-order models if needed

---

## Contributing

### Code Style

Follow **PEP 8** with these conventions:

```python
# Good: Descriptive names, type hints where useful
def process_hand_landmarks(
    landmarks: List[HandLandmark],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int]:
    """Get palm center position in frame coordinates."""
    # Comments explain WHY, not WHAT
    # Code should be clear enough for WHAT
    ...

# Avoid: Unclear abbreviations, missing context
def proc_lm(lm, fw, fh):
    x = sum([lm[i].x for i in [0, 5, 9, 13, 17]]) / 5
    ...
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>

type: feat|fix|docs|refactor|perf|test|chore
subject: lowercase, imperative, <50 chars
body: explain what and why, wrapped at 72 chars
footer: reference issues, breaking changes
```

Example:
```
feat: add left/right hand detection

- MediaPipe now detects handedness
- Update gesture detector to handle both hands
- Add hand_side parameter to all gesture methods

Fixes #45
BREAKING CHANGE: gesture.detect() signature changed
```

### Pull Request Process

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes, add tests
4. Run tests: `pytest tests/`
5. Check code style: `black . && pylint`
6. Commit: `git commit -m "..."`
7. Push: `git push origin feature/my-feature`
8. Create pull request with description

---

## Troubleshooting Development

### Import Errors

```python
# Ensure module paths are correct
# Add debugprint to identify which module fails
import sys
print(sys.path)  # Check search paths
```

### Gesture Not Detecting

```python
# Enable debug output
config.DEBUG_MODE = True
config.DEBUG_SHOW_GESTURE_SCORES = True

# Run app and observe gesture scores in logs
# Check if your gesture is appearing with high confidence
```

### Performance Degradation

```python
# Profile specific function
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Resources

- **MediaPipe Docs**: https://mediapipe.dev
- **OpenCV Docs**: https://docs.opencv.org
- **PyAutoGUI Docs**: https://pyautogui.readthedocs.io
- **Kalman Filters**: https://en.wikipedia.org/wiki/Kalman_filter
- **Python Best Practices**: https://pep8.org

---

**Happy coding! 💻**

For questions or issues, please open a GitHub issue or contact the maintainers.
