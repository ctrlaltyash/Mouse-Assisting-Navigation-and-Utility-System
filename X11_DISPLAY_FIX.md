# X11 Display Error Fix

## Problem
The application crashed when running in environments without X11 display access (headless systems, containers, WSL without X server, etc.):

```
Xlib.error.XauthError: ~/.Xauthority: [Errno 2] No such file or directory
```

This occurred because:
1. `pyautogui` was imported at module-level in `main.py`
2. `pyautogui` imports `mouseinfo` which requires X11 display connection
3. `tkinter` GUI also requires display capabilities

## Solution

### 1. **Lazy Loading PyAutoGUI** (`main.py`)
- Removed top-level `import pyautogui`
- Moved `pyautogui.size()` call inside `__init__` with try/except
- Falls back to default screen size (1920x1080) if X11 unavailable

### 2. **Graceful PyAutoGUI Import** (`action_handler.py`)
- Wrapped `import pyautogui` in try/except at module level
- Set `PYAUTOGUI_AVAILABLE` flag
- All action methods check this flag before executing
- Returns `False` if mouse/keyboard control unavailable

### 3. **Graceful Tkinter Import** (`ui/settings_gui.py`)
- Wrapped `import tkinter` in try/except
- Set `TKINTER_AVAILABLE` flag
- GUI skips initialization if tkinter unavailable
- Logs warning instead of crashing

## Behavior

### On Systems WITH X11 Display
- ✅ Full functionality (mouse, keyboard, GUI)
- Normal logging with no warnings

### On Headless Systems (no X11)
- ✅ Application starts successfully
- ⚠️ Shows warnings about unavailable features
- 🎥 Hand detection still works (OpenCV/MediaPipe)
- ❌ Mouse/keyboard actions disabled gracefully
- ❌ Settings GUI disabled gracefully

## Example Output

```
[WARNING] PyAutoGUI not available (X11 display may not be accessible)
[WARNING] Tkinter not available (display/X11 may not be accessible)
[INFO] ============================================================
[INFO] Initializing MANUS Hand Gesture Control System
[INFO] ============================================================
[INFO] ActionHandler initialized (mouse/keyboard control DISABLED - X11 not available)
[INFO] SettingsGUI cannot start: Tkinter not available
[INFO] MANUS System initialized successfully
```

## Testing

To test in headless mode:
```bash
# Simulate headless environment
unset DISPLAY
python main.py
```

The application should now start and log appropriate warnings instead of crashing.

## Files Modified

1. `main.py` - Lazy load pyautogui
2. `action_handler.py` - Graceful pyautogui import with feature flag
3. `ui/settings_gui.py` - Graceful tkinter import with feature flag
