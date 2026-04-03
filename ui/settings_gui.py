"""
Settings GUI for runtime configuration tuning.

Provides a tkinter-based interface to adjust parameters in real-time
without restarting the application.
"""

import threading
import config
from manus_logging.logger import get_logger

logger = get_logger()

# Try to import tkinter, gracefully handle missing display
try:
    import tkinter as tk
    from tkinter import Scale, Label, Button, Frame, HORIZONTAL, LEFT, TOP
    TKINTER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Tkinter not available (display/X11 may not be accessible): {e}")
    TKINTER_AVAILABLE = False
    # Define dummy classes to prevent import errors
    tk = None
    Scale = Label = Button = Frame = None
    HORIZONTAL = LEFT = TOP = None


class SettingsGUI:
    """
    Tkinter-based GUI for adjusting MANUS parameters at runtime.
    
    Runs in a separate thread and allows real-time tuning of:
    - Cursor smoothing
    - Detection confidence
    - Gesture thresholds
    - Camera resolution
    """
    
    def __init__(self, runtime_config):
        """
        Initialize settings GUI.
        
        Args:
            runtime_config: config.RuntimeConfig instance to update
        """
        self.runtime_config = runtime_config
        self.root = None
        self.thread = None
        self.is_running = False
        
        logger.debug("SettingsGUI initialized")
    
    def start(self):
        """Start GUI in separate thread."""
        if not TKINTER_AVAILABLE:
            logger.warning("SettingsGUI cannot start: Tkinter not available (display/X11 needed)")
            return
            
        if self.is_running:
            logger.warning("SettingsGUI already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("SettingsGUI started")
    
    def stop(self):
        """Stop GUI."""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                logger.warning(f"Error stopping GUI: {e}")
        logger.debug("SettingsGUI stopped")
    
    def _run(self):
        """Main GUI loop (runs in separate thread)."""
        try:
            self.root = tk.Tk()
            self.root.title("MANUS Settings")
            self.root.geometry(f"{config.UI_SETTINGS_WINDOW_WIDTH}x{config.UI_SETTINGS_WINDOW_HEIGHT}")
            self.root.resizable(False, False)
            
            # Create settings widgets
            self._create_widgets()
            
            # Update loop
            self.root.after(config.UI_SETTINGS_UPDATE_INTERVAL_MS, self._update_from_app)
            
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI error: {e}")
        finally:
            self.is_running = False
    
    def _create_widgets(self):
        """Create GUI controls."""
        
        # Title
        title_label = Label(self.root, text="MANUS Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Smoothing slider
        self._create_slider_group(
            "Cursor Smoothing",
            "smoothing",
            0.0, 1.0, 0.01,
            "Lower = smoother, Higher = snappier"
        )
        
        # Friction slider
        self._create_slider_group(
            "Cursor Friction",
            "friction",
            0.0, 1.0, 0.01,
            "Lower = more damping"
        )
        
        # Finger threshold slider
        self._create_slider_group(
            "Finger Threshold",
            "finger_threshold",
            0.05, 0.5, 0.01,
            "Higher = larger gesture tolerance"
        )
        
        # Detection confidence slider
        self._create_slider_group(
            "Detection Confidence",
            "detection_confidence",
            0.3, 1.0, 0.05,
            "Higher = stricter hand detection"
        )
        
        # Tracking confidence slider
        self._create_slider_group(
            "Tracking Confidence",
            "tracking_confidence",
            0.3, 1.0, 0.05,
            "Higher = more stable tracking"
        )
        
        # Camera resolution section
        res_frame = Frame(self.root)
        res_frame.pack(pady=10)
        
        res_label = Label(res_frame, text="Camera Resolution", font=("Arial", 10, "bold"))
        res_label.pack()
        
        self.res_var = tk.StringVar(value=f"{int(self.runtime_config.camera_width)}x{int(self.runtime_config.camera_height)}")
        
        resolutions = [
            ("320x240", 320, 240),
            ("640x480", 640, 480),
            ("1280x720", 1280, 720),
            ("1920x1080", 1920, 1080),
        ]
        
        for res_name, w, h in resolutions:
            btn = Button(res_frame, text=res_name,
                        command=lambda w=w, h=h: self._set_resolution(w, h),
                        width=10)
            btn.pack(side=LEFT, padx=5)
        
        # Status frame
        self.status_frame = Frame(self.root)
        self.status_frame.pack(pady=10)
        
        self.status_label = Label(self.status_frame, text="Status: Active", fg="green")
        self.status_label.pack()
        
        # Close button
        close_btn = Button(self.root, text="Close", command=self.stop, bg="red", fg="white")
        close_btn.pack(pady=10)
        
        logger.debug("GUI widgets created")
    
    def _create_slider_group(self, label_text, config_key, min_val, max_val, resolution, help_text):
        """
        Create a labeled slider group.
        
        Args:
            label_text: Display label
            config_key: Key in runtime_config to update
            min_val, max_val: Range
            resolution: Step size
            help_text: Tooltip text
        """
        frame = Frame(self.root)
        frame.pack(pady=5, padx=10, fill="x")
        
        label = Label(frame, text=label_text, width=20, anchor="w")
        label.pack(side=LEFT)
        
        current_value = getattr(self.runtime_config, config_key)
        
        slider = Scale(frame, from_=min_val, to=max_val, resolution=resolution,
                      orient=HORIZONTAL, length=150,
                      command=lambda v: self._on_slider_change(config_key, float(v)))
        slider.set(current_value)
        slider.pack(side=LEFT, padx=5)
        
        value_label = Label(frame, text=f"{current_value:.3f}", width=8)
        value_label.pack(side=LEFT)
        
        help_label = Label(frame, text=help_text, fg="gray", font=("Arial", 8))
        help_label.pack(side=TOP, anchor="w", padx=10)
        
        # Store references for updating
        if not hasattr(self, 'sliders'):
            self.sliders = {}
        self.sliders[config_key] = (slider, value_label)
    
    def _on_slider_change(self, config_key, value):
        """Handle slider change event."""
        setattr(self.runtime_config, config_key, value)
        logger.debug(f"Updated {config_key} = {value:.3f}")
    
    def _set_resolution(self, width, height):
        """Set camera resolution."""
        self.runtime_config.camera_width = width
        self.runtime_config.camera_height = height
        self.res_var.set(f"{width}x{height}")
        logger.info(f"Resolution changed to {width}x{height}")
    
    def _update_from_app(self):
        """Update GUI from application state (runs periodically)."""
        if not self.is_running or not self.root:
            return
        
        try:
            # Update slider values if they changed externally
            if hasattr(self, 'sliders'):
                for config_key, (slider, value_label) in self.sliders.items():
                    current_value = getattr(self.runtime_config, config_key)
                    slider.set(current_value)
                    value_label.config(text=f"{current_value:.3f}")
            
            # Update status
            status_text = "Active" if self.runtime_config.gesture_active else "Paused"
            status_color = "green" if self.runtime_config.gesture_active else "red"
            self.status_label.config(text=f"Status: {status_text}", fg=status_color)
            
            # Schedule next update
            self.root.after(config.UI_SETTINGS_UPDATE_INTERVAL_MS, self._update_from_app)
        except Exception as e:
            logger.error(f"GUI update error: {e}")
