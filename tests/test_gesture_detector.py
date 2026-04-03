"""
Unit tests for Gesture Detection module.
"""

import unittest
from unittest.mock import Mock, MagicMock
from gesture.gesture_detector import GestureDetector, distance


class MockLandmark:
    """Mock MediaPipe landmark for testing."""
    
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y


class TestGestureDetector(unittest.TestCase):
    """Test gesture detection functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = GestureDetector()
        self.landmarks = self._create_neutral_landmarks()
    
    def _create_neutral_landmarks(self):
        """Create neutral hand landmarks (open hand)."""
        # 21 landmarks in neutral position
        landmarks = [MockLandmark(0.5, 0.5) for _ in range(21)]
        return landmarks
    
    def _create_fist_landmarks(self):
        """Create fist gesture landmarks."""
        landmarks = self._create_neutral_landmarks()
        # Move all finger tips down (closed)
        landmarks[8].y = 0.4  # Index tip down
        landmarks[12].y = 0.4  # Middle tip down
        landmarks[16].y = 0.4  # Ring tip down
        landmarks[20].y = 0.4  # Pinky tip down
        return landmarks
    
    def _create_pointing_landmarks(self):
        """Create pointing gesture landmarks (index up, others down)."""
        landmarks = self._create_neutral_landmarks()
        landmarks[8].y = 0.3  # Index tip UP
        landmarks[12].y = 0.6  # Middle tip down
        landmarks[16].y = 0.6  # Ring tip down
        landmarks[20].y = 0.6  # Pinky tip down
        return landmarks
    
    def test_distance_function(self):
        """Test distance calculation."""
        lm1 = MockLandmark(0, 0)
        lm2 = MockLandmark(3, 4)
        
        d = distance(lm1, lm2)
        self.assertAlmostEqual(d, 5.0, places=1)  # 3-4-5 triangle
    
    def test_fist_detection(self):
        """Test fist gesture detection."""
        fist_lm = self._create_fist_landmarks()
        scores = self.detector.detect(fist_lm)
        
        # Fist should have highest confidence
        self.assertGreater(scores['fist'], 0.3)
    
    def test_pointing_detection(self):
        """Test pointing gesture detection."""
        pointing_lm = self._create_pointing_landmarks()
        scores = self.detector.detect(pointing_lm)
        
        # Pointing should have good confidence
        self.assertGreater(scores['pointing'], 0.3)
    
    def test_neutral_detection(self):
        """Test neutral hand (open) detection."""
        scores = self.detector.detect(self.landmarks)
        
        # Neutral should not strongly trigger any gesture
        max_score = max(v for k, v in scores.items() if k != 'idle')
        self.assertLess(max_score, 0.8)
    
    def test_all_gestures_detected(self):
        """Test that detector returns scores for all gestures."""
        scores = self.detector.detect(self.landmarks)
        
        expected_gestures = [
            'idle', 'fist', 'pointing', 'peace', 'right_click',
            'enter', 'thumbs_up', 'shaka', 'rock', 'three_finger_click'
        ]
        
        for gesture in expected_gestures:
            self.assertIn(gesture, scores)
            self.assertIsInstance(scores[gesture], (int, float))
            self.assertGreaterEqual(scores[gesture], 0.0)
            self.assertLessEqual(scores[gesture], 1.0)
    
    def test_scroll_detection_up(self):
        """Test scroll up gesture detection."""
        landmarks = self._create_neutral_landmarks()
        landmarks[8].y = 0.2  # Index tip UP (strong)
        landmarks[12].y = 0.7  # Middle down
        landmarks[16].y = 0.7  # Ring down
        landmarks[20].y = 0.7  # Pinky down
        
        direction, confidence = self.detector.detect_scroll(landmarks)
        
        self.assertEqual(direction, 'up')
        self.assertGreater(confidence, 0.5)
    
    def test_scroll_detection_down(self):
        """Test scroll down gesture detection."""
        landmarks = self._create_neutral_landmarks()
        landmarks[8].y = 0.8  # Index tip DOWN (strong)
        landmarks[12].y = 0.7  # Middle down
        landmarks[16].y = 0.7  # Ring down
        landmarks[20].y = 0.7  # Pinky down
        
        direction, confidence = self.detector.detect_scroll(landmarks)
        
        self.assertEqual(direction, 'down')
        self.assertGreater(confidence, 0.5)
    
    def test_scroll_detection_none(self):
        """Test no scroll gesture."""
        direction, confidence = self.detector.detect_scroll(self.landmarks)
        
        self.assertIsNone(direction)
        self.assertEqual(confidence, 0.0)
    
    def test_palm_position_scaling(self):
        """Test palm position coordinate scaling."""
        # Test with specific hand pose
        x, y = self.detector.get_hand_palm_position(self.landmarks, 640, 480)
        
        # Should return valid coordinates
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 640)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y, 480)


if __name__ == '__main__':
    unittest.main()
