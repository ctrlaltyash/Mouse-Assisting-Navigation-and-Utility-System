"""
Gesture validation module for reducing false positives.

Multi-frame confirmation, confidence scoring, and gesture deduplication.
"""

from collections import deque
import config
from manus_logging.logger import get_logger

logger = get_logger()


class GestureValidator:
    """
    Validates gestures over multiple frames to reduce false positives.
    
    Maintains a history of gesture detections and confirms only when
    a gesture appears consistently over N frames.
    """
    
    def __init__(self, confirmation_frames=None):
        """
        Initialize gesture validator.
        
        Args:
            confirmation_frames: Number of consecutive frames to confirm gesture.
        """
        self.confirmation_frames = confirmation_frames or config.GESTURE_CONFIRMATION_FRAMES
        self.gesture_history = deque(maxlen=self.confirmation_frames)
        self.last_confirmed_gesture = None
        self.confirmation_count = 0
    
    def update(self, current_gesture, confidence=1.0):
        """
        Update validation state with current frame's detected gesture.
        
        Args:
            current_gesture: String name of detected gesture (or None/Idle)
            confidence: Confidence score 0.0-1.0
        
        Returns:
            Tuple of (confirmed_gesture, confirmed_confidence) or (None, 0.0) if not confirmed
        """
        # Store current detection
        self.gesture_history.append({
            'gesture': current_gesture,
            'confidence': confidence
        })
        
        # Check if we have enough frames to confirm
        if len(self.gesture_history) < self.confirmation_frames:
            return None, 0.0
        
        # Check if all recent frames show the same gesture
        gestures = [g['gesture'] for g in self.gesture_history]
        first_gesture = gestures[0]
        
        # Special case: idle gesture doesn't need confirmation
        if first_gesture is None or first_gesture.lower() == 'idle':
            self.last_confirmed_gesture = None
            return None, 0.0
        
        # Check consistency
        if all(g == first_gesture for g in gestures):
            # Gesture is consistent - confirm it
            confidences = [g['confidence'] for g in self.gesture_history]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Only confirm if meets minimum confidence threshold
            if avg_confidence >= config.GESTURE_CONFIDENCE_MIN:
                # Check if this is a new gesture (edge-triggered)
                if self.last_confirmed_gesture != first_gesture:
                    self.last_confirmed_gesture = first_gesture
                    logger.debug(f"Gesture confirmed: {first_gesture} (confidence: {avg_confidence:.2f})")
                    return first_gesture, avg_confidence
        else:
            # Gesture changed or inconsistent
            self.last_confirmed_gesture = None
        
        return None, 0.0
    
    def reset(self):
        """Reset validation state."""
        self.gesture_history.clear()
        self.last_confirmed_gesture = None
        self.confirmation_count = 0
    
    def get_history_summary(self):
        """Get summary of recent gesture history (for debugging)."""
        if not self.gesture_history:
            return "No history"
        
        gestures = [g['gesture'] for g in self.gesture_history]
        return " → ".join([str(g)[:3] for g in gestures])


class CooldownTracker:
    """
    Tracks per-gesture cooldowns to prevent rapid re-triggering.
    """
    
    def __init__(self):
        """Initialize cooldown tracker."""
        self.cooldowns = {
            'click': config.COOLDOWN_CLICK,
            'right_click': config.COOLDOWN_RIGHT_CLICK,
            'enter': config.COOLDOWN_ENTER,
            'scroll': config.COOLDOWN_SCROLL,
            'action_hold': config.COOLDOWN_ACTION_HOLD,
        }
        self.last_trigger_time = {}
    
    def is_ready(self, action_name, current_time):
        """
        Check if an action is ready to trigger (cooldown expired).
        
        Args:
            action_name: Name of action (e.g., 'click', 'right_click')
            current_time: Current timestamp
        
        Returns:
            True if action can be triggered, False if in cooldown
        """
        if action_name not in self.cooldowns:
            logger.warning(f"Unknown action in cooldown tracker: {action_name}")
            return True
        
        cooldown = self.cooldowns[action_name]
        last_time = self.last_trigger_time.get(action_name, 0.0)
        
        if (current_time - last_time) >= cooldown:
            return True
        
        return False
    
    def trigger(self, action_name, current_time):
        """
        Mark an action as just triggered.
        
        Args:
            action_name: Name of action
            current_time: Current timestamp
        """
        self.last_trigger_time[action_name] = current_time
    
    def get_remaining_cooldown(self, action_name, current_time):
        """Get remaining cooldown time in seconds."""
        if action_name not in self.cooldowns:
            return 0.0
        
        cooldown = self.cooldowns[action_name]
        last_time = self.last_trigger_time.get(action_name, 0.0)
        remaining = cooldown - (current_time - last_time)
        
        return max(0.0, remaining)
    
    def reset(self):
        """Reset all cooldowns."""
        self.last_trigger_time.clear()
