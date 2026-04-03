"""
MANUS - Mouse Assisting Navigation and Utility System

Main orchestrator for the hand gesture control system.

Integrates all modules: camera capture, hand tracking, gesture detection,
action execution, and UI rendering.
"""

import cv2
import time
import numpy as np
import sys

import config
from manus_logging.logger import get_logger
from mediapipe_compat import MediaPipeHandsCompat, drawing_utils
from filters.kalman_filter import KalmanFilter
from gesture.gesture_detector import GestureDetector
from gesture.gesture_validator import GestureValidator, CooldownTracker
from actions.action_handler import ActionHandler
from ui.hud_renderer import HUDRenderer
from ui.settings_gui import SettingsGUI

logger = get_logger()


class MANUSSystem:
    """
    Main MANUS system coordinator.
    
    Manages camera capture, hand tracking, gesture detection, and action execution.
    """
    
    def __init__(self):
        """Initialize MANUS system."""
        logger.info("=" * 60)
        logger.info("Initializing MANUS Hand Gesture Control System")
        logger.info("=" * 60)
        
        # Camera
        self.cap = None
        self.frame_width = config.CAMERA_RESOLUTION_WIDTH
        self.frame_height = config.CAMERA_RESOLUTION_HEIGHT
        
        # MediaPipe
        self.mp_hands = MediaPipeHandsCompat
        self.hands = None
        self.draw = drawing_utils
        
        # Processing
        self.gesture_detector = GestureDetector()
        self.gesture_validator = GestureValidator()
        self.cooldown_tracker = CooldownTracker()
        self.kalman_filter = KalmanFilter()
        self.action_handler = ActionHandler()
        self.hud_renderer = HUDRenderer()
        self.settings_gui = None
        
        # State
        self.current_gesture = "Idle"
        self.gesture_confidence = 0.0
        self.hand_detected = False
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.clicking = False
        self.fist_hold_start = None
        self.prev_enter_state = False
        
        # Performance
        self.fps = 0.0
        self.frame_count = 0
        self.t_last = time.perf_counter()
        self.latency_history = []
        self.max_latency_history = 30
        
        # Safety - try to get screen size, use defaults if unavailable (e.g., headless/no X11)
        try:
            import pyautogui
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception as e:
            logger.warning(f"Could not detect screen size (X11 may not be available): {e}")
            # Use reasonable defaults for headless environments
            self.screen_width = 1920
            self.screen_height = 1080
        
        # Help flag
        self.show_help = False
        
        logger.info(f"Screen size: {self.screen_width}x{self.screen_height}")
        logger.info(f"Camera target: {self.frame_width}x{self.frame_height} @ {config.TARGET_FPS} FPS")
        
        self._initialize_camera()
        self._initialize_mediapipe()
        self._initialize_ui()
        
        logger.info("MANUS System initialized successfully")
    
    def _initialize_camera(self):
        """Initialize camera capture."""
        logger.info("Initializing camera...")
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Camera not accessible")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        
        # Enable optimization
        if config.ENABLE_CAMERA_OPTIMIZATION:
            cv2.setUseOptimized(True)
        
        # Read one frame to validate
        ret, test_frame = self.cap.read()
        if not ret:
            logger.error("Failed to read from camera")
            raise RuntimeError("Camera read error")
        
        self.frame_height, self.frame_width = test_frame.shape[:2]
        logger.info(f"Camera opened: {self.frame_width}x{self.frame_height}")
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe hand detection."""
        logger.info("Initializing MediaPipe...")
        
        self.hands = self.mp_hands(
            max_num_hands=config.MP_MAX_HANDS,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=config.runtime_config.detection_confidence,
            min_tracking_confidence=config.runtime_config.tracking_confidence,
        )
        
        logger.info("MediaPipe initialized")
    
    def _initialize_ui(self):
        """Initialize UI components."""
        logger.info("Initializing UI...")
        
        if config.UI_SETTINGS_ENABLED:
            self.settings_gui = SettingsGUI(config.runtime_config)
            self.settings_gui.start()
        
        logger.info("UI initialized")
    
    def run(self):
        """Main event loop."""
        logger.info("Starting main loop...")
        
        try:
            while True:
                # Capture frame
                t_frame_start = time.perf_counter()
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand detection
                results = self.hands.process(rgb_frame)
                
                # Update FPS
                self._update_fps()
                
                # Process gestures
                if results.multi_hand_landmarks:
                    self.hand_detected = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Detect palm position
                    palm_x, palm_y = self.gesture_detector.get_hand_palm_position(
                        hand_landmarks.landmark,
                        self.frame_width,
                        self.frame_height
                    )
                    
                    # Map to screen coordinates
                    screen_x = int(palm_x * self.screen_width / self.frame_width)
                    screen_y = int(palm_y * self.screen_height / self.frame_height)
                    
                    # Update Kalman filter
                    bounds = (0, self.screen_width, 0, self.screen_height)
                    self.cursor_x, self.cursor_y = self.kalman_filter.update(
                        [screen_x, screen_y],
                        ensure_bounds=bounds
                    )
                    
                    # Apply smoothing
                    self._apply_cursor_smoothing()
                    
                    # Move mouse
                    self.action_handler.move_mouse(self.cursor_x, self.cursor_y)
                    
                    # Detect gestures
                    self._process_gestures(hand_landmarks.landmark, time.time())
                    
                    # Render landmarks
                    connections = HUDRenderer.get_hand_connections()
                    frame = self.hud_renderer.render_hand_landmarks(
                        frame,
                        hand_landmarks.landmark,
                        connections
                    )
                    frame = self.hud_renderer.render_palm_center(frame, palm_x, palm_y)
                else:
                    self.hand_detected = False
                    self._handle_lost_hand()
                
                # Render HUD
                paused = not config.runtime_config.gesture_active
                frame = self.hud_renderer.render(
                    frame,
                    gesture_status=self.current_gesture,
                    fps=self.fps,
                    hand_detected=self.hand_detected,
                    paused=paused,
                    extra_info={
                        'Position': f"({int(self.cursor_x)}, {int(self.cursor_y)})",
                        'Confidence': f"{self.gesture_confidence:.2f}",
                    }
                )
                
                # Display frame
                cv2.imshow("MANUS - Hand Gesture Control", frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("Exit key pressed (ESC)")
                    break
                elif key == ord('h'):
                    self.show_help = not self.show_help
                
                # Measure latency
                t_frame_end = time.perf_counter()
                latency_ms = (t_frame_end - t_frame_start) * 1000
                self.latency_history.append(latency_ms)
                if len(self.latency_history) > self.max_latency_history:
                    self.latency_history.pop(0)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _update_fps(self):
        """Update FPS counter."""
        t_now = time.perf_counter()
        dt = t_now - self.t_last
        
        if dt > 0:
            self.fps = 1.0 / dt
        
        self.t_last = t_now
        self.frame_count += 1
    
    def _apply_cursor_smoothing(self):
        """Apply exponential smoothing to cursor position."""
        # The Kalman filter already provides smoothing, but we can add
        # additional exponential smoothing for extra stability
        smoothing = config.runtime_config.smoothing
        
        # Applied implicitly through Kalman filter adaptive noise
        # Additional smoothing could be added here if needed
        pass
    
    def _process_gestures(self, landmarks, current_time):
        """
        Process hand gestures.
        
        Args:
            landmarks: List of MediaPipe landmarks
            current_time: Current timestamp
        """
        if not config.runtime_config.gesture_active:
            return
        
        # Detect all gestures
        gesture_scores = self.gesture_detector.detect(landmarks)
        
        # Get highest-confidence gesture
        best_gesture = max(gesture_scores, key=gesture_scores.get)
        best_confidence = gesture_scores[best_gesture]
        
        # Filter low-confidence detections
        if best_confidence < config.GESTURE_CONFIDENCE_MIN:
            best_gesture = "Idle"
            best_confidence = 0.0
        
        # Validate gesture over multiple frames
        validated_gesture, validated_confidence = self.gesture_validator.update(
            best_gesture if best_confidence > 0.5 else None,
            best_confidence
        )
        
        self.current_gesture = validated_gesture or "Idle"
        self.gesture_confidence = validated_confidence
        
        # Execute actions for confirmed gestures
        if validated_gesture:
            self._execute_gesture_action(validated_gesture, landmarks, current_time)
    
    def _execute_gesture_action(self, gesture, landmarks, current_time):
        """
        Execute action for recognized gesture.
        
        Args:
            gesture: Gesture name string
            landmarks: List of landmarks
            current_time: Current timestamp
        """
        gesture_lower = gesture.lower()
        
        # Scroll
        if gesture_lower == "scroll":
            scroll_dir, _ = self.gesture_detector.detect_scroll(landmarks)
            if scroll_dir and self.cooldown_tracker.is_ready('scroll', current_time):
                if scroll_dir == 'up':
                    self.action_handler.press_key('pageup')
                else:
                    self.action_handler.press_key('pagedown')
                self.cooldown_tracker.trigger('scroll', current_time)
        
        # Fist (drag/click)
        elif gesture_lower == "fist":
            if not self.clicking and self.cooldown_tracker.is_ready('action_hold', current_time):
                self.action_handler.mouse_down()
                self.clicking = True
                logger.debug("Click/drag started")
        
        # Right click
        elif gesture_lower == "right_click":
            if self.cooldown_tracker.is_ready('right_click', current_time):
                self.action_handler.click_right()
                self.cooldown_tracker.trigger('right_click', current_time)
        
        # Three finger click
        elif gesture_lower == "three_finger_click":
            if self.cooldown_tracker.is_ready('click', current_time):
                self.action_handler.click_left()
                self.cooldown_tracker.trigger('click', current_time)
        
        # Enter key
        elif gesture_lower == "enter":
            if not self.prev_enter_state and self.cooldown_tracker.is_ready('enter', current_time):
                self.action_handler.press_key('enter')
                self.cooldown_tracker.trigger('enter', current_time)
                self.prev_enter_state = True
            self.prev_enter_state = True
        else:
            self.prev_enter_state = False
        
        # Thumbs up (set center)
        if gesture_lower == "thumbs_up":
            center_pos = self.action_handler.get_mouse_position()
            logger.info(f"Center set to: {center_pos}")
        
        # Shaka (reset center)
        if gesture_lower == "shaka":
            logger.info("Center reset")
        
        # Rock/easter egg
        if gesture_lower == "rock" and config.ENABLE_SPECIAL_GESTURES and config.ENABLE_RICKROLL:
            self.action_handler.open_browser("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    def _handle_lost_hand(self):
        """Handle when hand tracking is lost."""
        self.clicking = False
        self.prev_enter_state = False
        self.fist_hold_start = None
        self.gesture_validator.reset()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")
        
        try:
            if self.settings_gui:
                self.settings_gui.stop()
        except Exception as e:
            logger.warning(f"Error stopping settings GUI: {e}")
        
        try:
            if self.clicking:
                self.action_handler.mouse_up()
        except Exception as e:
            logger.warning(f"Error releasing mouse: {e}")
        
        try:
            if self.cap:
                self.cap.release()
        except Exception as e:
            logger.warning(f"Error closing camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Error destroying windows: {e}")
        
        # Log final stats
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Average FPS: {self.fps:.1f}")
        if self.latency_history:
            avg_latency = np.mean(self.latency_history)
            logger.info(f"Average latency: {avg_latency:.1f}ms")
        logger.info(f"Total actions: {self.action_handler.get_action_count()}")
        logger.info("MANUS System shutdown")


def main():
    """Entry point."""
    try:
        system = MANUSSystem()
        system.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
