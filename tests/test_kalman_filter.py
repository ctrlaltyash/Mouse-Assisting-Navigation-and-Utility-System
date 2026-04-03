"""
Unit tests for Kalman Filter module.
"""

import unittest
import numpy as np
from filters.kalman_filter import KalmanFilter


class TestKalmanFilter(unittest.TestCase):
    """Test KalmanFilter accuracy and convergence."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kf = KalmanFilter()
    
    def test_initialization(self):
        """Test filter initialization."""
        self.assertFalse(self.kf.is_initialized)
        self.assertEqual(self.kf.updates_count, 0)
    
    def test_first_update(self):
        """Test first measurement initializes filter."""
        measurement = [100, 200]
        x, y = self.kf.update(measurement)
        
        self.assertTrue(self.kf.is_initialized)
        self.assertAlmostEqual(x, 100, places=1)
        self.assertAlmostEqual(y, 200, places=1)
    
    def test_constant_position(self):
        """Test filter with constant position (no motion)."""
        position = [100, 200]
        
        for _ in range(10):
            x, y = self.kf.update(position)
        
        # Should converge to correct position
        self.assertAlmostEqual(x, 100, places=0)
        self.assertAlmostEqual(y, 200, places=0)
    
    def test_linear_motion(self):
        """Test filter tracking linear motion."""
        # Generate linear motion: +10 pixels per frame horizontally
        positions = [[100 + i*5, 200] for i in range(10)]
        
        results = []
        for pos in positions:
            x, y = self.kf.update(pos)
            results.append((x, y))
        
        # Filter should predict reasonable positions
        final_x, final_y = results[-1]
        self.assertGreater(final_x, 130)  # Should have moved in right direction
        self.assertEqual(self.kf.updates_count, 10)
    
    def test_bounds_clamping(self):
        """Test position clamping to bounds."""
        bounds = (50, 150, 50, 250)
        
        # Point outside bounds
        x, y = self.kf.update([200, 300], ensure_bounds=bounds)
        
        self.assertGreaterEqual(x, bounds[0])
        self.assertLessEqual(x, bounds[1])
        self.assertGreaterEqual(y, bounds[2])
        self.assertLessEqual(y, bounds[3])
    
    def test_velocity_tracking(self):
        """Test velocity estimation."""
        # Constant velocity motion
        positions = [[100 + i*10, 200] for i in range(5)]
        
        for pos in positions:
            self.kf.update(pos)
        
        vx, vy = self.kf.get_velocity()
        # Should estimate positive x-velocity
        self.assertGreater(vx, 0)
    
    def test_reset(self):
        """Test filter reset."""
        # Do some updates
        for i in range(5):
            self.kf.update([100 + i, 200])
        
        self.assertTrue(self.kf.is_initialized)
        
        # Reset
        self.kf.reset()
        
        self.assertFalse(self.kf.is_initialized)
        self.assertEqual(self.kf.updates_count, 0)


if __name__ == '__main__':
    unittest.main()
