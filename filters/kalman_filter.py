"""
Enhanced Kalman Filter for cursor smoothing.

Features:
- Constant velocity model
- Adaptive noise based on hand velocity
- Better initialization and convergence
- Position bounds checking
"""

import numpy as np
import config


class KalmanFilter:
    """Enhanced 2D Kalman Filter for hand tracking cursor smoothing."""
    
    def __init__(self, process_noise=None, measurement_noise=None, adaptive_noise=True):
        """
        Initialize Kalman Filter.
        
        Args:
            process_noise: Process noise covariance (Q). Default from config.
            measurement_noise: Measurement noise covariance (R). Default from config.
            adaptive_noise: Whether to adapt noise based on hand velocity.
        """
        self.process_noise = process_noise or config.KF_PROCESS_NOISE
        self.measurement_noise = measurement_noise or config.KF_MEASUREMENT_NOISE
        self.adaptive_noise = adaptive_noise
        
        # State vector: [x, y, vx, vy]^T (position + velocity)
        self.x = np.zeros((4, 1))
        self.x_prev = np.zeros((4, 1))
        
        # State transition matrix (constant velocity model)
        # x_{k+1} = F * x_k (position += velocity * dt)
        self.F = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],  # vx' = vx (constant)
            [0, 0, 0, 1],  # vy' = vy (constant)
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Initialize covariance matrices
        self.Q = self.process_noise * np.eye(4, dtype=np.float32)
        self.R = self.measurement_noise * np.eye(2, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0  # Initial uncertainty
        
        # Filter state
        self.is_initialized = False
        self.updates_count = 0
        self.velocity_history = []
        self.max_velocity_history = 100
    
    def _calculate_velocity_magnitude(self):
        """Calculate current velocity magnitude."""
        vx = self.x[2, 0]
        vy = self.x[3, 0]
        return np.sqrt(vx**2 + vy**2)
    
    def _adapt_noise(self, velocity_mag):
        """
        Adapt process and measurement noise based on hand velocity.
        
        Fast moving hands (high velocity) -> trust measurements more (reduce R)
        Slow moving hands (low velocity) -> trust motion model more (reduce Q)
        """
        if not self.adaptive_noise or velocity_mag is None:
            return
        
        # Normalize velocity (assuming max velocity ~500 pixels/frame)
        normalized_velocity = min(velocity_mag / 500.0, 1.0)
        
        # Adaptive Q: lower when moving fast (trust motion model)
        q_factor = 0.1 + (1 - normalized_velocity) * 0.9
        self.Q = self.process_noise * np.eye(4, dtype=np.float32) * q_factor
        
        # Adaptive R: lower when moving fast (trust measurements)
        r_factor = 0.5 + (1 - normalized_velocity) * 0.5
        self.R = self.measurement_noise * np.eye(2, dtype=np.float32) * r_factor
    
    def update(self, measurement, ensure_bounds=None):
        """
        Update filter with new measurement.
        
        Args:
            measurement: [x, y] observed position
            ensure_bounds: Tuple of (min_x, max_x, min_y, max_y) to clamp position
        
        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        z = np.reshape(measurement, (2, 1)).astype(np.float32)
        
        # First frame: initialize state
        if not self.is_initialized:
            self.x[0:2, 0] = z[:, 0]
            self.x[2:4, 0] = 0.0  # Zero velocity initially
            self.is_initialized = True
            self.x_prev = self.x.copy()
            return z[0, 0], z[1, 0]
        
        # --- PREDICTION ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # --- ADAPTATION ---
        velocity_mag = self._calculate_velocity_magnitude()
        self._adapt_noise(velocity_mag)
        
        # --- CORRECTION ---
        innovation = z - self.H @ x_pred  # Measurement residual
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
            K = P_pred @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            # Singular matrix: use pseudo-inverse
            K = P_pred @ self.H.T @ np.linalg.pinv(S)
        
        # Update state
        self.x_prev = self.x.copy()
        self.x = x_pred + K @ innovation
        
        # Update covariance
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ P_pred
        
        # Ensure bounds if specified
        if ensure_bounds:
            min_x, max_x, min_y, max_y = ensure_bounds
            self.x[0, 0] = np.clip(self.x[0, 0], min_x, max_x)
            self.x[1, 0] = np.clip(self.x[1, 0], min_y, max_y)
        
        # Track velocity history
        self.velocity_history.append(velocity_mag)
        if len(self.velocity_history) > self.max_velocity_history:
            self.velocity_history.pop(0)
        
        self.updates_count += 1
        
        return float(self.x[0, 0]), float(self.x[1, 0])
    
    def reset(self):
        """Reset filter to uninitialized state."""
        self.x = np.zeros((4, 1))
        self.x_prev = np.zeros((4, 1))
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.is_initialized = False
        self.updates_count = 0
        self.velocity_history = []
    
    def get_state(self):
        """Get current filter state: (x, y, vx, vy)."""
        return self.x.flatten().tolist()
    
    def get_position(self):
        """Get current estimated position."""
        return float(self.x[0, 0]), float(self.x[1, 0])
    
    def get_velocity(self):
        """Get current estimated velocity."""
        return float(self.x[2, 0]), float(self.x[3, 0])
    
    def get_estimated_covariance(self):
        """Get estimation covariance (uncertainty in estimate)."""
        # Return diagonal (variance) for position components
        return float(self.P[0, 0]), float(self.P[1, 1])
    
    def get_average_velocity(self):
        """Get average velocity magnitude from recent history."""
        if not self.velocity_history:
            return 0.0
        return float(np.mean(self.velocity_history))
