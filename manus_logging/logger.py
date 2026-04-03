"""
Logging module for MANUS Hand Gesture Control System.

Provides structured logging to both console and file with rotation support.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

import config


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS and config.LOG_TO_CONSOLE:
            record.levelname = self.COLORS[levelname] + levelname + self.RESET
        return super().format(record)


class MANUSLogger:
    """Centralized logging for MANUS system."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger("MANUS")
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        if not config.LOG_ENABLED:
            self.logger.addHandler(logging.NullHandler())
            return
        
        # Create formatters
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            fmt='[%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                config.LOG_FILE,
                maxBytes=config.LOG_FILE_SIZE_MB * 1024 * 1024,
                backupCount=config.LOG_FILE_BACKUPS
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
        
        # Console handler
        if config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def log_gesture(self, gesture_name, confidence=1.0):
        """Log detected gesture."""
        if config.DEBUG_SHOW_GESTURE_SCORES:
            self.debug(f"Gesture detected: {gesture_name} (confidence: {confidence:.2f})")
        else:
            self.debug(f"Gesture detected: {gesture_name}")
    
    def log_performance(self, fps, latency_ms, hand_detected):
        """Log performance metrics."""
        if config.LOG_PERFORMANCE_METRICS:
            hand_status = "✓" if hand_detected else "✗"
            self.debug(f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Hand: {hand_status}")
    
    def log_action(self, action_type, details=""):
        """Log executed action."""
        msg = f"Action: {action_type}"
        if details:
            msg += f" - {details}"
        self.info(msg)
    
    def log_error_with_context(self, error_msg, context=None):
        """Log error with optional context."""
        msg = f"ERROR: {error_msg}"
        if context:
            msg += f" | Context: {context}"
        self.error(msg)


# Global logger instance
logger = MANUSLogger().logger


def get_logger():
    """Get the global MANUS logger."""
    return MANUSLogger().logger
