"""
Action handler module for executing mouse and keyboard commands.

Provides safe, logged execution of user actions with error handling.
"""

import webbrowser
import threading
import config
from manus_logging.logger import get_logger

logger = get_logger()

# Try to import pyautogui, gracefully handle X11/display issues
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception as e:
    logger.warning(f"PyAutoGUI not available (X11 display may not be accessible): {e}")
    PYAUTOGUI_AVAILABLE = False


class ActionHandler:
    """
    Centralized handler for all mouse and keyboard actions.
    
    Provides safety checks, logging, and thread-safe execution.
    """
    
    def __init__(self):
        """Initialize action handler."""
        self.last_action = None
        self.action_count = 0
        self.pyautogui_available = PYAUTOGUI_AVAILABLE
        
        # Disable pyautogui safety pause for faster execution (if available)
        if PYAUTOGUI_AVAILABLE:
            pyautogui.PAUSE = 0
        
        if self.pyautogui_available:
            logger.info("ActionHandler initialized (with mouse/keyboard control)")
        else:
            logger.warning("ActionHandler initialized (mouse/keyboard control DISABLED - X11 not available)")
    
    def move_mouse(self, x, y):
        """
        Move mouse to position with bounds checking.
        
        Args:
            x, y: Target screen coordinates
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            # Clamp to screen bounds
            if config.MOUSE_BOUNDS_CHECK:
                screen_width, screen_height = pyautogui.size()
                x = max(config.MOUSE_EDGE_DEAD_ZONE, min(screen_width - config.MOUSE_EDGE_DEAD_ZONE, x))
                y = max(config.MOUSE_EDGE_DEAD_ZONE, min(screen_height - config.MOUSE_EDGE_DEAD_ZONE, y))
            
            pyautogui.moveTo(int(x), int(y), duration=0)
            self.last_action = ('move', x, y)
            return True
        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False
    
    def click_left(self):
        """
        Execute left mouse click.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.click(button='left')
            self.action_count += 1
            logger.log_action('left_click')
            self.last_action = ('click_left',)
            return True
        except Exception as e:
            logger.error_with_context("Left click failed", str(e))
            return False
    
    def click_right(self):
        """
        Execute right mouse click.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.click(button='right')
            self.action_count += 1
            logger.log_action('right_click')
            self.last_action = ('click_right',)
            return True
        except Exception as e:
            logger.error_with_context("Right click failed", str(e))
            return False
    
    def click_middle(self):
        """Execute middle mouse click."""
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.click(button='middle')
            self.action_count += 1
            logger.log_action('middle_click')
            self.last_action = ('click_middle',)
            return True
        except Exception as e:
            logger.error_with_context("Middle click failed", str(e))
            return False
    
    def mouse_down(self):
        """Press and hold mouse button (for dragging)."""
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.mouseDown()
            logger.debug("Mouse button down")
            self.last_action = ('mouse_down',)
            return True
        except Exception as e:
            logger.error_with_context("Mouse down failed", str(e))
            return False
    
    def mouse_up(self):
        """Release held mouse button."""
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.mouseUp()
            logger.debug("Mouse button up")
            self.action_count += 1
            logger.log_action('drag_complete')
            self.last_action = ('mouse_up',)
            return True
        except Exception as e:
            logger.error_with_context("Mouse up failed", str(e))
            return False
    
    def scroll(self, direction, amount=3):
        """
        Scroll the mouse wheel.
        
        Args:
            direction: 'up' or 'down'
            amount: Number of scroll units
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            if config.SCROLL_DIRECTION_INVERSE:
                direction = 'down' if direction == 'up' else 'up'
            
            scroll_amount = amount if direction == 'up' else -amount
            scroll_amount *= int(config.SCROLL_VELOCITY_MULTIPLIER)
            
            pyautogui.scroll(scroll_amount)
            self.action_count += 1
            logger.log_action(f'scroll_{direction}')
            self.last_action = ('scroll', direction)
            return True
        except Exception as e:
            logger.error_with_context("Scroll failed", str(e))
            return False
    
    def press_key(self, key):
        """
        Press a keyboard key.
        
        Args:
            key: Key name (e.g., 'enter', 'escape', 'pageup', 'pagedown')
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.press(key)
            self.action_count += 1
            logger.log_action(f'key_press', f"key={key}")
            self.last_action = ('press_key', key)
            return True
        except Exception as e:
            logger.error_with_context(f"Key press failed ({key})", str(e))
            return False
    
    def type_text(self, text):
        """
        Type text (slowly to avoid issues).
        
        Args:
            text: String to type
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pyautogui_available:
            return False
            
        try:
            pyautogui.typewrite(text, interval=0.05)
            self.action_count += 1
            logger.log_action('type_text', f"length={len(text)}")
            self.last_action = ('type_text', text)
            return True
        except Exception as e:
            logger.error_with_context("Text typing failed", str(e))
            return False
    
    def open_browser(self, url):
        """
        Open URL in web browser.
        
        Args:
            url: URL to open
        
        Returns:
            True if successful, False otherwise
        """
        try:
            def _open():
                webbrowser.open(url)
            
            if config.ASYNC_BROWSER_OPEN:
                # Open asynchronously on separate thread
                thread = threading.Thread(target=_open, daemon=True)
                thread.start()
            else:
                _open()
            
            self.action_count += 1
            logger.log_action('open_browser', url[:50])
            self.last_action = ('open_browser', url)
            return True
        except Exception as e:
            logger.error_with_context("Browser open failed", str(e))
            return False
    
    def get_mouse_position(self):
        """Get current mouse position."""
        try:
            return pyautogui.position()
        except Exception as e:
            logger.error(f"Failed to get mouse position: {e}")
            return None
    
    def get_screen_size(self):
        """Get screen dimensions."""
        try:
            return pyautogui.size()
        except Exception as e:
            logger.error(f"Failed to get screen size: {e}")
            return None
    
    def get_action_count(self):
        """Get total number of actions executed."""
        return self.action_count
    
    def reset_action_count(self):
        """Reset action counter."""
        self.action_count = 0
        logger.debug("Action counter reset")
