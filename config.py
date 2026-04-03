"""
Configuration module for MANUS Hand Gesture Control System.

All tunable parameters are centralized here for easy production-level adjustment
without modifying code. Parameters are organized by category.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = DATA_DIR / "hand_tracking_log.txt"
CONFIG_FILE = DATA_DIR / "settings.json"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# CAMERA / VIDEO CAPTURE
# ============================================================================
CAMERA_INDEX = 0  # Primary camera (0 = default)
CAMERA_RESOLUTION_WIDTH = 640  # Auto-adjust based on system performance
CAMERA_RESOLUTION_HEIGHT = 480
TARGET_FPS = 30  # Target frame rate
ENABLE_CAMERA_OPTIMIZATION = True  # Use OpenCV optimizations

# ============================================================================
# MEDIAPIPE HAND DETECTION
# ============================================================================
MP_MAX_HANDS = 1  # Single hand tracking (one is for palm-based control)
MP_MODEL_COMPLEXITY = 1  # 0=lite, 1=full (tradeoff: speed vs accuracy)
MP_DETECTION_CONFIDENCE = 0.6  # Minimum confidence to detect hand
MP_TRACKING_CONFIDENCE = 0.6  # Minimum confidence to track landmarks

# ============================================================================
# KALMAN FILTER SETTINGS (Cursor Smoothing)
# ============================================================================
KF_PROCESS_NOISE = 1e-4  # How much we trust the motion model (lower = trust model more)
KF_MEASUREMENT_NOISE = 1.0  # How much we trust measurements (higher = ignore noisy measurements)
KF_ADAPTIVE_NOISE = True  # Adapt noise based on hand velocity
KF_POSITION_SCALE = 1.0  # Scale factor for position uncertainty

# ============================================================================
# CURSOR MOVEMENT & SMOOTHING
# ============================================================================
CURSOR_SMOOTHING = 0.35  # Exponential smoothing factor (0.0=no smoothing, 1.0=full smoothing)
CURSOR_FRICTION = 0.8  # Velocity dampening (lower = more damping)
CURSOR_SMOOTH_FACTOR = 0.1  # Momentum transfer (physics simulation)
CURSOR_MOVEMENT_THRESHOLD = 5  # Minimum pixel change to move cursor
CURSOR_DEAD_ZONE = 5  # Pixel tolerance at screen edges

# ============================================================================
# HAND TRACKING & NORMALIZATION
# ============================================================================
HAND_BASE_SCALE_NORMALIZE = True  # Scale gestures relative to hand size
HAND_PALM_MAPPING_ROI_X_MIN = 0.05
HAND_PALM_MAPPING_ROI_X_MAX = 0.95
HAND_PALM_MAPPING_ROI_Y_MIN = 0.05
HAND_PALM_MAPPING_ROI_Y_MAX = 0.95
HAND_TRACKING_LOSS_TOLERANCE = 5  # Frames to wait before resetting on tracking loss

# ============================================================================
# GESTURE DETECTION THRESHOLDS (Scaled by hand size)
# ============================================================================
# Threshold multiplier for finger position detection (relative to hand scale)
GESTURE_FINGER_UP_THRESHOLD = 0.2  # Y-distance to consider finger "up"
GESTURE_FINGER_DOWN_THRESHOLD = 0.2  # Y-distance to consider finger "down"
GESTURE_THUMB_FOLD_THRESHOLD = 0.35  # Horizontal distance for thumb-folded check
GESTURE_FIST_CLOSURE_THRESHOLD = 0.18  # How tightly fist must close
GESTURE_FIST_RELEASE_THRESHOLD = 0.15  # Minimum fingers up to release fist

# Multi-frame validation (reduce false positives)
GESTURE_CONFIRMATION_FRAMES = 3  # Gesture must be valid for N consecutive frames
GESTURE_CONFIDENCE_MIN = 0.5  # Minimum confidence score (0.0-1.0)

# ============================================================================
# GESTURE COOLDOWNS (seconds) - Per-gesture, not global
# ============================================================================
COOLDOWN_CLICK = 0.3  # Left click cooldown
COOLDOWN_RIGHT_CLICK = 1.0  # Right click cooldown
COOLDOWN_ENTER = 1.0  # Enter key cooldown
COOLDOWN_SCROLL = 0.5  # Scroll action cooldown
COOLDOWN_ACTION_HOLD = 0.5  # Minimum hold time before triggering

# Fist toggle for pausing gestures
FIST_TOGGLE_HOLD_TIME = 1.8  # Seconds to hold fist to toggle pause state

# ============================================================================
# SCROLL BEHAVIOR
# ============================================================================
SCROLL_ACTION = "pageup_pagedown"  # "wheel" for mouse wheel, "pageup_pagedown" for Page keys
SCROLL_DIRECTION_INVERSE = False  # Invert up/down scroll direction
SCROLL_VELOCITY_MULTIPLIER = 1.0  # How many scroll units per gesture

# ============================================================================
# CLICK BEHAVIOR
# ============================================================================
CLICK_TYPE_ON_FIST = "drag"  # "click" for single, "drag" for mouse down/up
CLICK_BUTTON_DEFAULT = "left"  # "left", "right", "middle"

# ============================================================================
# UI / HUD RENDERING
# ============================================================================
UI_HUD_ENABLED = True  # Show HUD overlay on camera feed
UI_HUD_POSITION_X = 10  # Pixel offset from left
UI_HUD_POSITION_Y = 30  # Pixel offset from top
UI_HUD_FONT = None  # None = OpenCV default (cv2.FONT_HERSHEY_SIMPLEX)
UI_HUD_FONT_SCALE = 0.7  # Text scale
UI_HUD_FONT_THICKNESS = 2  # Text thickness in pixels

# HUD Colors (BGR format for OpenCV)
UI_COLOR_ACTIVE = (0, 255, 0)  # Green when active
UI_COLOR_PAUSED = (0, 0, 255)  # Red when paused
UI_COLOR_HIGHLIGHT = (0, 255, 255)  # Cyan for highlights
UI_COLOR_ERROR = (0, 0, 255)  # Red for errors
UI_COLOR_TEXT = (255, 255, 255)  # White text

# Settings GUI
UI_SETTINGS_ENABLED = True  # Show tkinter settings window
UI_SETTINGS_WINDOW_WIDTH = 400
UI_SETTINGS_WINDOW_HEIGHT = 500
UI_SETTINGS_UPDATE_INTERVAL_MS = 500  # Refresh rate of GUI updates

# Hand landmark visualization
UI_DRAW_LANDMARKS = True  # Draw hand skeleton on camera feed
UI_DRAW_PALM_CENTER = False  # Draw estimated palm center point

# ============================================================================
# LOGGING & DIAGNOSTICS
# ============================================================================
LOG_ENABLED = True
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE_SIZE_MB = 10  # Max log file size before rotation
LOG_FILE_BACKUPS = 3  # Number of backup log files to keep
LOG_TO_CONSOLE = True  # Also print to console
LOG_PERFORMANCE_METRICS = True  # Log FPS, latency, etc.

# ============================================================================
# PERFORMANCE & OPTIMIZATION
# ============================================================================
ENABLE_ADAPTIVE_QUALITY = True  # Auto-reduce resolution if FPS drops
ADAPTIVE_FPS_MIN_TARGET = 25  # Minimum FPS before quality reduction
ADAPTIVE_FPS_REDUCE_FACTOR = 0.8  # Reduce resolution by this factor

MEASURE_PIPELINE_LATENCY = True  # Measure end-to-end latency
ASYNC_BROWSER_OPEN = True  # Open web browser asynchronously

# ============================================================================
# ACCESSIBILITY & SAFETY
# ============================================================================
MOUSE_BOUNDS_CHECK = True  # Keep cursor inside screen bounds
MOUSE_EDGE_DEAD_ZONE = 10  # Pixels from screen edge to ignore
GESTURE_ENABLE_TOGGLE = True  # Allow fist-hold to pause/resume

# Special gestures
ENABLE_SPECIAL_GESTURES = True  # Thumbs up (center set), Shaka (center reset), Rock (rickroll)
ENABLE_RICKROLL = False  # Disable rickroll by default (easter egg)

# ============================================================================
# DEBUG MODE
# ============================================================================
DEBUG_MODE = False  # Enable debug prints and verbose output
DEBUG_SHOW_LANDMARK_CONFIDENCE = False  # Show confidence scores for landmarks
DEBUG_SHOW_GESTURE_SCORES = False  # Show gesture confidence scores
DEBUG_SIMULATE_HAND = False  # Simulate hand tracking for testing (no camera)

# ============================================================================
# Runtime State (can be modified by settings GUI)
# ============================================================================
class RuntimeConfig:
    """Mutable configuration that can be changed during runtime."""
    
    def __init__(self):
        self.gesture_active = True  # Start with gestures enabled
        self.smoothing = CURSOR_SMOOTHING
        self.friction = CURSOR_FRICTION
        self.detection_confidence = MP_DETECTION_CONFIDENCE
        self.tracking_confidence = MP_TRACKING_CONFIDENCE
        self.finger_threshold = GESTURE_FINGER_UP_THRESHOLD
        self.camera_width = CAMERA_RESOLUTION_WIDTH
        self.camera_height = CAMERA_RESOLUTION_HEIGHT
    
    def to_dict(self):
        """Export current state as dictionary."""
        return self.__dict__.copy()
    
    def from_dict(self, state_dict):
        """Restore state from dictionary."""
        self.__dict__.update(state_dict)


# Global runtime config instance
runtime_config = RuntimeConfig()
