"""
HUD (Heads-Up Display) rendering for the camera feed.

Displays gesture status, FPS, hand tracking info, and debug data.
"""

import cv2
import numpy as np
import config
from manus_logging.logger import get_logger

logger = get_logger()


class HUDRenderer:
    """
    Renders HUD elements on the camera frame.
    
    Displays gesture status, FPS, hand tracking state, and optional debug info.
    """
    
    def __init__(self):
        """Initialize HUD renderer."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.UI_HUD_FONT_SCALE
        self.font_thickness = config.UI_HUD_FONT_THICKNESS
        self.position_x = config.UI_HUD_POSITION_X
        self.position_y = config.UI_HUD_POSITION_Y
        self.line_height = 30
        
        logger.debug("HUDRenderer initialized")
    
    def render(self, frame, gesture_status=None, fps=0.0, hand_detected=False,
               extra_info=None, paused=False):
        """
        Render HUD elements on frame.
        
        Args:
            frame: OpenCV frame (numpy array)
            gesture_status: Current gesture name string
            fps: Current FPS value
            hand_detected: Whether hand is currently detected
            extra_info: Dictionary of additional debug info to display
            paused: Whether gesture control is paused
        
        Returns:
            Modified frame with HUD
        """
        if not config.UI_HUD_ENABLED:
            return frame
        
        frame = frame.copy()
        y_offset = self.position_y
        
        # Determine color based on state
        if paused:
            color = config.UI_COLOR_PAUSED
            status = "PAUSED"
        else:
            color = config.UI_COLOR_ACTIVE
            status = "ACTIVE"
        
        # Main status line
        if gesture_status is None:
            gesture_status = "Idle"
        
        hud_text = f"Gesture: {gesture_status} | FPS: {fps:.1f} | Status: {status}"
        self._put_text(frame, hud_text, self.position_x, y_offset, color)
        y_offset += self.line_height
        
        # Hand detection status
        hand_status = "✓ Hand Detected" if hand_detected else "✗ No Hand"
        hand_color = config.UI_COLOR_ACTIVE if hand_detected else config.UI_COLOR_PAUSED
        self._put_text(frame, hand_status, self.position_x, y_offset, hand_color)
        y_offset += self.line_height
        
        # Extra debug info
        if extra_info and config.DEBUG_MODE:
            for key, value in extra_info.items():
                info_text = f"{key}: {value}"
                self._put_text(frame, info_text, self.position_x, y_offset, config.UI_COLOR_TEXT)
                y_offset += self.line_height
        
        return frame
    
    def _put_text(self, frame, text, x, y, color):
        """Helper to put text with background for readability."""
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, color, self.font_thickness)
    
    def render_hand_landmarks(self, frame, landmarks, connections=None):
        """
        Draw hand skeleton (landmarks and connections) on frame.
        
        Args:
            frame: OpenCV frame
            landmarks: List of MediaPipe landmarks
            connections: List of (start_idx, end_idx) tuples for connections
        
        Returns:
            Modified frame with landmarks drawn
        """
        if not config.UI_DRAW_LANDMARKS or landmarks is None:
            return frame
        
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw connections (bones)
        if connections:
            for start_idx, end_idx in connections:
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                
                cv2.line(frame, start_pt, end_pt, config.UI_COLOR_HIGHLIGHT, 2)
        
        # Draw landmarks (joints) as circles
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Color based on landmark type
            if idx == 0:
                color = (255, 0, 0)  # Blue for wrist
            elif idx in [5, 9, 13, 17]:  # MCPs
                color = (0, 255, 0)  # Green for knuckles
            else:
                color = (0, 255, 255)  # Cyan for other joints
            
            cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame
    
    def render_palm_center(self, frame, x, y, radius=5):
        """
        Draw estimated palm center point.
        
        Args:
            frame: OpenCV frame
            x, y: Palm center coordinates
            radius: Circle radius
        
        Returns:
            Modified frame
        """
        if not config.UI_DRAW_PALM_CENTER:
            return frame
        
        frame = frame.copy()
        cv2.circle(frame, (int(x), int(y)), radius, config.UI_COLOR_HIGHLIGHT, -1)
        cv2.circle(frame, (int(x), int(y)), radius + 1, config.UI_COLOR_TEXT, 1)
        
        return frame
    
    def render_debug_overlay(self, frame, debug_info):
        """
        Render debug information overlay.
        
        Args:
            frame: OpenCV frame
            debug_info: Dictionary with debug keys/values
        
        Returns:
            Modified frame
        """
        if not config.DEBUG_MODE or debug_info is None:
            return frame
        
        frame = frame.copy()
        y_offset = 150  # Start below main HUD
        
        for key, value in debug_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), self.font, 0.5, config.UI_COLOR_TEXT, 1)
            y_offset += 20
        
        return frame
    
    @staticmethod
    def get_hand_connections():
        """
        Get MediaPipe hand landmark connections.
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        # 21 landmarks connected as hand skeleton
        connections = [
            # Palm
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            # Cross connections for palm
            (5, 9), (9, 13), (13, 17),
        ]
        return connections
