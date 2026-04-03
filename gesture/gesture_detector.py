"""
Gesture detection module for hand pose recognition.

Detects various gestures like pointing, fist, thumbs up, rock, etc.
with confidence scoring and adaptive thresholds.
"""

import math
import numpy as np
import config
from manus_logging.logger import get_logger

logger = get_logger()


# ============================================================================
# Helper Functions
# ============================================================================

def distance(landmark_a, landmark_b):
    """
    Calculate Euclidean distance between two normalized landmarks.
    
    Args:
        landmark_a, landmark_b: MediaPipe landmark objects with x, y attributes
    
    Returns:
        Distance as float
    """
    dx = landmark_a.x - landmark_b.x
    dy = landmark_a.y - landmark_b.y
    return math.hypot(dx, dy)


def get_hand_base_scale(landmarks):
    """
    Calculate hand size using distances from wrist to MCPs (knuckles).
    
    Uses normalized units (0..1 range). Allows adaptive thresholds based on hand size.
    
    Args:
        landmarks: List of MediaPipe landmarks (21 points)
    
    Returns:
        Average distance in normalized units
    """
    wrist = landmarks[0]
    # MCPs (middle joints) of fingers: thumb(2), index(5), middle(9), ring(13), pinky(17)
    mcp_indices = [5, 9, 13, 17]  # Exclude thumb MCP for stability
    
    distances = [distance(wrist, landmarks[i]) for i in mcp_indices]
    base_scale = sum(distances) / len(distances)
    
    return max(base_scale, 1e-6)  # Avoid division by zero


def finger_is_up(landmarks, tip_idx, pip_idx, threshold):
    """
    Determine if a finger is extended upward.
    
    Args:
        landmarks: List of landmarks
        tip_idx: Index of finger tip
        pip_idx: Index of PIP joint (middle joint)
        threshold: Y-distance threshold (relative to hand scale)
    
    Returns:
        Boolean
    """
    return landmarks[tip_idx].y < landmarks[pip_idx].y - threshold


def finger_is_down(landmarks, tip_idx, pip_idx, threshold):
    """Determine if a finger is folded downward."""
    return landmarks[tip_idx].y > landmarks[pip_idx].y + threshold


def get_confidence_score(landmarks, gesture_checks):
    """
    Calculate confidence score for a gesture based on how well it matches criteria.
    
    Args:
        landmarks: List of landmarks
        gesture_checks: List of boolean checks that should be True
    
    Returns:
        Confidence score 0.0-1.0
    """
    if not gesture_checks:
        return 0.0
    
    matches = sum(gesture_checks)
    return float(matches / len(gesture_checks))


# ============================================================================
# Gesture Detection Functions
# ============================================================================

class GestureDetector:
    """
    Main gesture detection class.
    
    Handles detection of all supported hand gestures with confidence scoring.
    """
    
    def __init__(self):
        """Initialize gesture detector."""
        self.finger_up_threshold = config.GESTURE_FINGER_UP_THRESHOLD
        self.finger_down_threshold = config.GESTURE_FINGER_DOWN_THRESHOLD
        self.fist_threshold = config.GESTURE_FIST_CLOSURE_THRESHOLD
    
    def detect(self, landmarks):
        """
        Detect all gestures in given hand landmarks.
        
        Args:
            landmarks: List of 21 MediaPipe landmarks
        
        Returns:
            Dictionary mapping gesture names to confidence scores
        """
        base_scale = get_hand_base_scale(landmarks)
        
        results = {
            'idle': 0.5,  # Default low confidence
            'fist': self._detect_fist(landmarks, base_scale),
            'pointing': self._detect_pointing(landmarks, base_scale),
            'peace': self._detect_peace(landmarks, base_scale),
            'right_click': self._detect_right_click(landmarks, base_scale),
            'enter': self._detect_enter(landmarks, base_scale),
            'thumbs_up': self._detect_thumbs_up(landmarks, base_scale),
            'shaka': self._detect_shaka(landmarks, base_scale),
            'rock': self._detect_rock(landmarks, base_scale),
            'three_finger_click': self._detect_three_finger_click(landmarks, base_scale),
        }
        
        return results
    
    def detect_scroll(self, landmarks):
        """
        Detect scroll gesture (index finger up/down with others down).
        
        Returns:
            Tuple of (direction, confidence) where direction is 'up', 'down', or None
        """
        base_scale = get_hand_base_scale(landmarks)
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        # Check if index is up and others are down
        idx_up = finger_is_up(landmarks, 8, 6, thr)
        mid_down = finger_is_down(landmarks, 12, 10, thr / 2)
        ring_down = finger_is_down(landmarks, 16, 14, thr / 2)
        pink_down = finger_is_down(landmarks, 20, 18, thr / 2)
        
        if idx_up and mid_down and ring_down and pink_down:
            if landmarks[8].y < landmarks[6].y - 1.2 * thr:
                return 'up', 0.8  # Strong up
            elif landmarks[8].y < landmarks[6].y - 0.5 * thr:
                return 'up', 0.6  # Weak up
        
        # Check if index is down and others are also down
        idx_down = finger_is_down(landmarks, 8, 6, thr)
        if idx_down and mid_down and ring_down and pink_down:
            if landmarks[8].y > landmarks[6].y + 1.2 * thr:
                return 'down', 0.8  # Strong down
            elif landmarks[8].y > landmarks[6].y + 0.5 * thr:
                return 'down', 0.6  # Weak down
        
        return None, 0.0
    
    def _detect_fist(self, landmarks, base_scale):
        """All fingers tightly folded."""
        thr = config.GESTURE_FIST_CLOSURE_THRESHOLD * base_scale
        
        checks = [
            finger_is_down(landmarks, 8, 6, thr),
            finger_is_down(landmarks, 12, 10, thr),
            finger_is_down(landmarks, 16, 14, thr),
            finger_is_down(landmarks, 20, 18, thr),
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_pointing(self, landmarks, base_scale):
        """Index finger up, others down."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),
            finger_is_down(landmarks, 12, 10, thr),
            finger_is_down(landmarks, 16, 14, thr),
            finger_is_down(landmarks, 20, 18, thr),
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_peace(self, landmarks, base_scale):
        """Index and middle fingers up, others down."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),  # Index up
            finger_is_up(landmarks, 12, 10, thr),  # Middle up
            finger_is_down(landmarks, 16, 14, thr),  # Ring down
            finger_is_down(landmarks, 20, 18, thr),  # Pinky down
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_right_click(self, landmarks, base_scale):
        """Index and middle up, ring and pinky down."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),  # Index up
            finger_is_up(landmarks, 12, 10, thr),  # Middle up
            finger_is_down(landmarks, 16, 14, thr),  # Ring down
            finger_is_down(landmarks, 20, 18, thr),  # Pinky down
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_enter(self, landmarks, base_scale):
        """Index, middle, ring up, pinky down, thumb folded."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        thumb_folded = abs(landmarks[4].x - landmarks[2].x) < config.GESTURE_THUMB_FOLD_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),  # Index up
            finger_is_up(landmarks, 12, 10, thr),  # Middle up
            finger_is_up(landmarks, 16, 14, thr),  # Ring up
            finger_is_down(landmarks, 20, 18, thr / 2),  # Pinky down
            thumb_folded,
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_thumbs_up(self, landmarks, base_scale):
        """Thumb up, all fingers down."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 4, 3, thr),  # Thumb up
            finger_is_down(landmarks, 8, 6, thr),  # Index down
            finger_is_down(landmarks, 12, 10, thr),  # Middle down
            finger_is_down(landmarks, 16, 14, thr),  # Ring down
            finger_is_down(landmarks, 20, 18, thr),  # Pinky down
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_shaka(self, landmarks, base_scale):
        """Thumb and pinky up, other fingers down (hang loose/peace sign variant)."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 4, 3, thr),  # Thumb up
            finger_is_up(landmarks, 20, 18, thr),  # Pinky up
            finger_is_down(landmarks, 8, 6, thr),  # Index down
            finger_is_down(landmarks, 12, 10, thr),  # Middle down
            finger_is_down(landmarks, 16, 14, thr),  # Ring down
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_rock(self, landmarks, base_scale):
        """Index and pinky up, others down (rock on gesture)."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),  # Index up
            finger_is_up(landmarks, 20, 18, thr),  # Pinky up
            not finger_is_up(landmarks, 12, 10, thr),  # Middle NOT up
            not finger_is_up(landmarks, 16, 14, thr),  # Ring NOT up
        ]
        return get_confidence_score(landmarks, checks)
    
    def _detect_three_finger_click(self, landmarks, base_scale):
        """Index, middle, ring up, pinky down."""
        thr = config.GESTURE_FINGER_UP_THRESHOLD * base_scale
        
        checks = [
            finger_is_up(landmarks, 8, 6, thr),  # Index up
            finger_is_up(landmarks, 12, 10, thr),  # Middle up
            finger_is_up(landmarks, 16, 14, thr),  # Ring up
            finger_is_down(landmarks, 20, 18, thr / 2),  # Pinky down
        ]
        return get_confidence_score(landmarks, checks)
    
    def get_hand_palm_position(self, landmarks, frame_width, frame_height):
        """
        Get estimated palm center position in frame coordinates.
        
        Args:
            landmarks: List of 21 landmarks
            frame_width, frame_height: Camera frame dimensions
        
        Returns:
            Tuple of (x, y) in frame coordinates (pixels)
        """
        # Average position of wrist (0) and MCPs (5, 9, 13, 17)
        x_coords = [landmarks[i].x for i in [0, 5, 9, 13, 17]]
        y_coords = [landmarks[i].y for i in [0, 5, 9, 13, 17]]
        
        x_avg = sum(x_coords) / len(x_coords)
        y_avg = sum(y_coords) / len(y_coords)
        
        # Apply ROI scaling to normalize hand position mapping
        x_scaled = min(max((x_avg - config.HAND_PALM_MAPPING_ROI_X_MIN) / 
                          (config.HAND_PALM_MAPPING_ROI_X_MAX - config.HAND_PALM_MAPPING_ROI_X_MIN),
                          0.0), 1.0)
        y_scaled = min(max((y_avg - config.HAND_PALM_MAPPING_ROI_Y_MIN) / 
                          (config.HAND_PALM_MAPPING_ROI_Y_MAX - config.HAND_PALM_MAPPING_ROI_Y_MIN),
                          0.0), 1.0)
        
        x_pixel = int(x_scaled * frame_width)
        y_pixel = int(y_scaled * frame_height)
        
        return x_pixel, y_pixel
