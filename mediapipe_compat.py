"""
MediaPipe compatibility layer for Tasks API.

Provides a wrapper around MediaPipe Tasks API (0.10+) to maintain
backwards compatibility with code written for the Solutions API (pre-0.10).
"""

import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from manus_logging.logger import get_logger

logger = get_logger()


class Landmark:
    """Wrapper to mimic old Solutions API Landmark object."""
    
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class NormalizedLandmarkList:
    """Wrapper to mimic old Solutions API NormalizedLandmarkList."""
    
    def __init__(self, landmarks):
        self.landmark = landmarks


class HandLandmarksResult:
    """Wrapper to mimic old Solutions API results."""
    
    def __init__(self, hand_landmarks_list, multi_handedness=None):
        self.multi_hand_landmarks = hand_landmarks_list
        self.multi_handedness = multi_handedness


class MediaPipeHandsCompat:
    """
    Compatibility wrapper for MediaPipe hand detection.
    
    Wraps the new Tasks API (HandLandmarker) to provide the old Solutions API interface.
    """
    
    def __init__(self, max_num_hands=2, model_complexity=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            model_complexity: 0 (lite) or 1 (full)
            min_detection_confidence: Detection confidence threshold
            min_tracking_confidence: Tracking confidence threshold
        """
        logger.info("Initializing MediaPipe HandLandmarker (Tasks API)...")
        
        # Try to find the model file in common locations
        import os
        import tempfile
        import urllib.request
        
        # Model file names
        # MediaPipe 0.10+ generally provides a single `hand_landmarker.task` common model.
        model_name = "hand_landmarker.task"
        model_url = f"https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/{model_name}"
        
        # Check common mediapipe package locations
        import mediapipe
        mediapipe_path = os.path.dirname(mediapipe.__file__)
        
        possible_paths = [
            os.path.join(mediapipe_path, 'tasks', 'python', 'components', 'processors', 'data', model_name),
            os.path.join(mediapipe_path, 'tasks', 'python', 'vision', 'models', model_name),
            os.path.join(mediapipe_path, 'models', model_name),
            os.path.join(os.path.expanduser('~'), '.cache', 'mediapipe', model_name),
            model_name,
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model at: {path}")
                break
        
        # If not found, try to download or use a fallback
        if not model_path:
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'mediapipe')
            os.makedirs(cache_dir, exist_ok=True)
            cache_model_path = os.path.join(cache_dir, model_name)
            
            if os.path.exists(cache_model_path):
                model_path = cache_model_path
                logger.info(f"Using cached model: {cache_model_path}")
            else:
                logger.warning(f"Model file not found locally. Attempting to download from {model_url}")
                try:
                    logger.info(f"Downloading {model_name}...")
                    urllib.request.urlretrieve(model_url, cache_model_path)
                    model_path = cache_model_path
                    logger.info(f"Model downloaded to: {cache_model_path}")
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    # Fall back to bundled lite model if available
                    fallback_lite = os.path.join(mediapipe_path, 'tasks', 'python', 'components', 'processors', 'data', 'hand_landmarker.task')
                    if os.path.exists(fallback_lite):
                        model_path = fallback_lite
                        logger.warning(f"Using fallback bundled model: {fallback_lite}")
                    else:
                        # Last resort: use model name and hope MediaPipe can find it
                        model_path = model_name
                        logger.warning(f"Using model name as-is: {model_name}")
        
        logger.info(f"Using model: {model_path}")
        
        # Create base options
        base_options = BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        try:
            self.hand_landmarker = HandLandmarker.create_from_options(options)
            self.mp_image = None
            logger.info("MediaPipe HandLandmarker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create HandLandmarker: {e}")
            logger.error("\n" + "=" * 70)
            logger.error("MODEL FILES NOT FOUND")
            logger.error("=" * 70)
            logger.error("The MediaPipe hand landmarker model files are required but not found.")
            logger.error("\nTo fix this:")
            logger.error("  1. Run: python download_models.py")
            logger.error("  2. Or manually download from:")
            logger.error("     https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker")
            logger.error("  3. Place in: ~/.cache/mediapipe/")
            logger.error("=" * 70)
            raise RuntimeError(f"MediaPipe model not available: {e}")
    
    def process(self, image_rgb):
        """
        Process an RGB image and detect hand landmarks.
        
        Args:
            image_rgb: RGB image as numpy array (H x W x 3)
        
        Returns:
            Result object with multi_hand_landmarks attribute
        """
        try:
            # Convert BGR to RGB if needed and create MediaPipe Image
            if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
                logger.error("Invalid image format for hand detection")
                return HandLandmarksResult([])
            
            # Create MediaPipe Image object
            from mediapipe import Image, ImageFormat
            
            # Ensure uint8 format
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
            
            mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
            
            # Run hand landmarker
            detection_result = self.hand_landmarker.detect(mp_image)
            
            # Convert results to old API format
            hand_landmarks_list = []
            
            if detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    if landmarks is None:
                        continue
                    # Convert to old format
                    landmark_list = []
                    for landmark in landmarks:
                        if landmark is None:
                            continue

                        # Tasks API may return partial values; ignore invalid points.
                        x = getattr(landmark, 'x', None)
                        y = getattr(landmark, 'y', None)
                        z = getattr(landmark, 'z', 0.0)
                        vis = getattr(landmark, 'visibility', 1.0)

                        if x is None or y is None:
                            continue

                        try:
                            compat_landmark = Landmark(
                                x=float(x),
                                y=float(y),
                                z=float(z) if z is not None else 0.0,
                                visibility=float(vis) if vis is not None else 1.0,
                            )
                        except (TypeError, ValueError) as e:
                            logger.debug(f"Skipping invalid landmark value: {e}")
                            continue

                        landmark_list.append(compat_landmark)

                    if landmark_list:
                        hand_landmarks_list.append(NormalizedLandmarkList(landmark_list))

            # Return wrapped result
            return HandLandmarksResult(hand_landmarks_list)
        
        except Exception as e:
            logger.error(f"Error processing hand detection: {e}")
            return HandLandmarksResult([])
    
    def close(self):
        """Close the hand landmarker."""
        if self.hand_landmarker:
            self.hand_landmarker.close()


# Create a simple namespace for drawing utilities (only used for visualization)
class drawing_utils:
    """Stub for drawing utilities - not used in headless mode."""
    
    @staticmethod
    def draw_landmarks(*args, **kwargs):
        """Stub - does nothing."""
        pass
    
    @staticmethod
    def draw_landmarks_on_image(*args, **kwargs):
        """Stub - does nothing."""
        pass


# Export the compatibility classes
__all__ = [
    'MediaPipeHandsCompat',
    'Landmark',
    'NormalizedLandmarkList',
    'HandLandmarksResult',
    'drawing_utils',
]
