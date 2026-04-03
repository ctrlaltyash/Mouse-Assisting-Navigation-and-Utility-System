"""
Test fixtures and mock data for MANUS testing.
"""

from unittest.mock import Mock


class MockLandmark:
    """Mock MediaPipe landmark for testing."""
    
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_neutral_hand(scale=1.0):
    """
    Create neutral/open hand pose landmarks.
    
    Args:
        scale: Relative hand size (1.0 = full size)
    
    Returns:
        List of 21 MockLandmark objects
    """
    # Neutral positions for each of 21 joints
    positions = [
        (0.5, 0.5),    # 0: Wrist
        (0.5, 0.4),    # 1-4: Thumb
        (0.45, 0.3),
        (0.4, 0.2),
        (0.35, 0.1),
        (0.6, 0.4),    # 5-8: Index finger
        (0.65, 0.3),
        (0.7, 0.2),
        (0.75, 0.1),
        (0.7, 0.4),    # 9-12: Middle finger
        (0.75, 0.3),
        (0.8, 0.2),
        (0.85, 0.1),
        (0.8, 0.4),    # 13-16: Ring finger
        (0.85, 0.3),
        (0.9, 0.2),
        (0.95, 0.1),
        (0.9, 0.4),    # 17-20: Pinky finger
        (0.95, 0.3),
        (1.0, 0.2),
        (1.05, 0.1),
    ]
    
    return [MockLandmark(x * scale, y * scale) for x, y in positions]


def create_fist_hand(scale=1.0):
    """Create closed fist pose."""
    landmarks = create_neutral_hand(scale)
    # Move all finger tips close to palm
    for i in [4, 8, 12, 16, 20]:
        landmarks[i].y = 0.6  # Down
    return landmarks


def create_pointing_hand(scale=1.0):
    """Create pointing gesture (index up, others down)."""
    landmarks = create_neutral_hand(scale)
    landmarks[8].y = 0.05   # Index tip UP
    landmarks[12].y = 0.65  # Middle down
    landmarks[16].y = 0.65  # Ring down
    landmarks[20].y = 0.65  # Pinky down
    return landmarks


def create_peace_hand(scale=1.0):
    """Create peace gesture (index+middle up)."""
    landmarks = create_neutral_hand(scale)
    landmarks[8].y = 0.05   # Index up
    landmarks[12].y = 0.05  # Middle up
    landmarks[16].y = 0.65  # Ring down
    landmarks[20].y = 0.65  # Pinky down
    return landmarks


def create_thumbs_up_hand(scale=1.0):
    """Create thumbs up gesture."""
    landmarks = create_neutral_hand(scale)
    landmarks[4].y = 0.05  # Thumb UP
    # All fingers down
    for i in [8, 12, 16, 20]:
        landmarks[i].y = 0.65
    return landmarks


def create_shaka_hand(scale=1.0):
    """Create shaka gesture (thumb+pinky up)."""
    landmarks = create_neutral_hand(scale)
    landmarks[4].y = 0.05   # Thumb UP
    landmarks[20].y = 0.05  # Pinky UP
    # Other fingers down
    for i in [8, 12, 16]:
        landmarks[i].y = 0.65
    return landmarks


def create_moving_hand_sequence(num_frames=10):
    """
    Create sequence of hand moving across screen.
    
    Useful for testing cursor tracking.
    
    Args:
        num_frames: Number of frames in sequence
    
    Returns:
        List of landmark sequences
    """
    sequence = []
    for i in range(num_frames):
        # Hand moves linearly from left (0.2) to right (0.8)
        x_offset = 0.2 + (0.6 * i / num_frames)
        landmarks = create_neutral_hand()
        # Shift all landmarks
        for lm in landmarks:
            lm.x += x_offset
            lm.x = min(max(lm.x, 0.0), 1.0)  # Clamp to [0, 1]
        sequence.append(landmarks)
    return sequence


def create_gesture_sequence(gesture_type, num_frames=10):
    """
    Create sequence of repeated gesture.
    
    Args:
        gesture_type: 'fist', 'pointing', 'peace', etc.
        num_frames: Number of frames
    
    Returns:
        List of landmark sequences
    """
    gestures = {
        'fist': create_fist_hand,
        'pointing': create_pointing_hand,
        'peace': create_peace_hand,
        'thumbs_up': create_thumbs_up_hand,
        'shaka': create_shaka_hand,
    }
    
    if gesture_type not in gestures:
        raise ValueError(f"Unknown gesture: {gesture_type}")
    
    gesture_func = gestures[gesture_type]
    return [gesture_func() for _ in range(num_frames)]


def create_noisy_hand_sequence(base_landmarks, noise_amount=0.05, num_variants=10):
    """
    Create sequence with added noise (realistic hand jitter).
    
    Args:
        base_landmarks: Reference landmark set
        noise_amount: Standard deviation of noise (0-1 scale)
        num_variants: Number of noisy variants
    
    Returns:
        List of noisy landmark sequences
    """
    import random
    
    sequence = []
    for _ in range(num_variants):
        variant = []
        for base_lm in base_landmarks:
            lm = MockLandmark(
                x=base_lm.x + random.gauss(0, noise_amount),
                y=base_lm.y + random.gauss(0, noise_amount),
                z=base_lm.z + random.gauss(0, noise_amount),
            )
            # Clamp to valid range
            lm.x = min(max(lm.x, 0.0), 1.0)
            lm.y = min(max(lm.y, 0.0), 1.0)
            lm.z = min(max(lm.z, -1.0), 1.0)
            variant.append(lm)
        sequence.append(variant)
    
    return sequence


# Preset hand poses for common tests
HAND_POSES = {
    'neutral': create_neutral_hand,
    'fist': create_fist_hand,
    'pointing': create_pointing_hand,
    'peace': create_peace_hand,
    'thumbs_up': create_thumbs_up_hand,
    'shaka': create_shaka_hand,
}
