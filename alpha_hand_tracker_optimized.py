import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser
import math
import numpy as np

class SimpleKalman:
    def __init__(self, process_noise=1e-4, measurement_noise=1):
        # State: [x, y, vx, vy]^T  (position + velocity)
        self.x = np.zeros((4, 1))

        # State transition (constant velocity model)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix (we only measure x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process and measurement covariance
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)

        # Estimate covariance (how sure we are)
        self.P = np.eye(4)

    def update(self, z):
        """z is the observed [x, y] column vector."""
        z = np.reshape(z, (2, 1))

        # --- Prediction ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Correction ---
        y = z - self.H @ self.x                                # Innovation
        S = self.H @ self.P @ self.H.T + self.R                # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)               # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[0, 0], self.x[1, 0]  # filtered x, y

# -------- Performance Tweaks --------
cv2.setUseOptimized(True)          # OpenCV internal optimizations
pyautogui.PAUSE = 0                # No extra delay after each call

# -------- Screen Size --------
wScr, hScr = pyautogui.size()

# -------- Video / Mediapipe Setup --------
cap = cv2.VideoCapture(0)
# You can lower the camera resolution to reduce CPU load:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
draw = mp.solutions.drawing_utils

# -------- State --------
clicking = False
click_cooldown = 0.0
right_click_time = 0.0
enter_press_time = 0.0
scroll_time = 0.0
last_rickroll_time = 0.0
kf = SimpleKalman()

# For the Physics
cursor_x, cursor_y = 0.0, 0.0
velocity_x, velocity_y = 0.0, 0.0
smooth_factor = 0.1
friction = 0.8



SCROLL_INTERVAL = 0.5
RIGHT_CLICK_COOLDOWN = 1.0
ENTER_COOLDOWN = 1.0
CLICK_HOLD_DELAY = 0.5

# Movement smoothing + threshold (less jitter)
MOVE_THRESHOLD = 5             # pixels: minimum change to move mouse
SMOOTHING = 0.35               # 0..1; higher = snappier, lower = smoother
last_mouse_x, last_mouse_y = None, None

center_x_offset = 0
center_y_offset = 0
custom_center = None

# Safety toggle (enable/disable gestures) by holding a full fist
GESTURE_ACTIVE = True          # Start active
fist_hold_start = None         # Time when fist started being held
FIST_TOGGLE_SECONDS = 1.8      # Hold time to toggle

# --------- Helpers ---------
def dist(a, b):
    """Euclidean distance between two normalized landmarks (x,y)."""
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

def get_base_scale(lm):
    """A hand-size scale using distances from wrist (0) to MCPs (5,9,13,17)."""
    wrist = lm[0]
    mcp_ids = [5, 9, 13, 17]
    d = [dist(wrist, lm[i]) for i in mcp_ids]
    return sum(d) / len(d)  # normalized units (0..1)

def finger_up(lm, tip, pip, thr):
    """Is finger 'up' by comparing vertical y with a threshold offset."""
    return lm[tip].y < lm[pip].y - thr

def finger_down(lm, tip, pip, thr):
    return lm[tip].y > lm[pip].y + thr

def thumb_folded(lm, base_scale):
    # Thumb folded if tip close to MCP horizontally
    
    return abs(lm[4].x - lm[2].x) < 0.35 * base_scale

def get_palm_position_scaled(lm, w, h):
    """
    Map averaged palm/wrist area to image coordinates (0..w, 0..h),
    then we scale to screen. This keeps your original ROI mapping.
    """
    x_raw = (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5
    y_raw = (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5
    x_scaled = min(max((x_raw - 0.05) / 0.9, 0.0), 1.0)
    y_scaled = min(max((y_raw - 0.05) / 0.9, 0.0), 1.0)
    return int(x_scaled * w), int(y_scaled * h)

def detect_scroll(lm, base_scale):
    """
    Scroll when index is clearly up/down and others relaxed down.
    Using scale-relative thresholds for distance invariance.
    """
    thr = 0.2 * base_scale
    idx_up = finger_up(lm, 8, 6, thr)
    mid_down = finger_down(lm, 12, 10, thr / 2)
    ring_down = finger_down(lm, 16, 14, thr / 2)
    pink_down = finger_down(lm, 20, 18, thr / 2)

    if idx_up and mid_down and ring_down and pink_down:
        # Strong up (extra margin) -> PageUp
        if lm[8].y < lm[6].y - 1.2 * thr:
            return "up"
    elif finger_down(lm, 8, 6, thr) and mid_down and ring_down and pink_down:
        # Strong down -> PageDown
        if lm[8].y > lm[6].y + 1.2 * thr:
            return "down"
    return None

def is_full_fist(lm, base_scale):
    thr = 0.18 * base_scale
    return (
        finger_down(lm, 8, 6, thr)
        and finger_down(lm, 12, 10, thr)
        and finger_down(lm, 16, 14, thr)
        and finger_down(lm, 20, 18, thr)
    )

def is_fist_released(lm, base_scale):
    thr = 0.15 * base_scale
    up_fingers = sum([
        finger_up(lm, 8, 6, thr),
        finger_up(lm, 12, 10, thr),
        finger_up(lm, 16, 14, thr),
        finger_up(lm, 20, 18, thr),
    ])
    return up_fingers >= 3

def is_right_click(lm, base_scale):
    thr = 0.16 * base_scale
    index_up = finger_up(lm, 8, 6, thr)
    middle_up = finger_up(lm, 12, 10, thr)
    ring_down_ = finger_down(lm, 16, 14, thr)
    pinky_down_ = finger_down(lm, 20, 18, thr)
    return index_up and middle_up and ring_down_ and pinky_down_

def is_enter_gesture(lm, base_scale):
    thr = 0.16 * base_scale
    index_up = finger_up(lm, 8, 6, thr)
    middle_up = finger_up(lm, 12, 10, thr)
    ring_up = finger_up(lm, 16, 14, thr)
    pinky_down_ = finger_down(lm, 20, 18, thr / 2)
    return index_up and middle_up and ring_up and pinky_down_ and thumb_folded(lm, base_scale)

def rock_and_roll(lm, base_scale):
    thr = 0.16 * base_scale
    index_up = finger_up(lm, 8, 6, thr)
    pinky_up = finger_up(lm, 20, 18, thr)
    middle_up = finger_up(lm, 12, 10, thr)
    ring_up = finger_up(lm, 16, 14, thr)
    return index_up and pinky_up and (not middle_up) and (not ring_up)

def is_three_finger_click(lm, base_scale):
    thr = 0.16 * base_scale
    index_up = finger_up(lm, 8, 6, thr)
    middle_up = finger_up(lm, 12, 10, thr)
    ring_up = finger_up(lm, 16, 14, thr)
    pinky_down_ = finger_down(lm, 20, 18, thr / 2)  # Pinky stays down
    return index_up and middle_up and ring_up and pinky_down_

def is_thumbs_up(lm, base_scale):
    thr = 0.15 * base_scale
    thumb_up = finger_up(lm, 4, 3, thr)
    index_down = finger_down(lm, 8, 6, thr)
    middle_down = finger_down(lm, 12, 10, thr)
    ring_down = finger_down(lm, 16, 14, thr)
    pinky_down = finger_down(lm, 20, 18, thr)
    return thumb_up and index_down and middle_down and ring_down and pinky_down

def is_shaka(lm, base_scale):
    thr = 0.15 * base_scale
    thumb_up = finger_up(lm, 4, 3, thr)
    pinky_up = finger_up(lm, 20, 18, thr)
    index_down = finger_down(lm, 8, 6, thr)
    middle_down = finger_down(lm, 12, 10, thr)
    ring_down = finger_down(lm, 16, 14, thr)
    return thumb_up and pinky_up and index_down and middle_down and ring_down


# Track previous enter state (edge-triggered)
prev_enter_state = False

# FPS calc
t_last = time.perf_counter()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Downscale for speed (2x faster typical), then flip for mirror view
    H, W = frame.shape[:2]
    small = cv2.resize(frame, (W // 1, H // 1))
    small = cv2.flip(small, 1)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "Idle" if GESTURE_ACTIVE else "Paused"
    now = time.time()

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        draw.draw_landmarks(small, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        base_scale = max(get_base_scale(lm), 1e-6)  # avoid zero

        # --- Safety toggle: hold full fist to toggle active state ---
        if is_full_fist(lm, base_scale):
            if fist_hold_start is None:
                fist_hold_start = now
            elif (now - fist_hold_start) >= FIST_TOGGLE_SECONDS:
                GESTURE_ACTIVE = not GESTURE_ACTIVE
                fist_hold_start = None
        else:
            fist_hold_start = None

        # --- Cursor control (only if active) ---
        # Use the downscaled frame dims for palm mapping
        if GESTURE_ACTIVE:
            palm_x, palm_y = get_palm_position_scaled(lm, W // 2, H // 2)
            screen_x = int(palm_x * wScr / (W // 2))
            screen_y = int(palm_y * hScr / (H // 2))

            # Exponential smoothing + movement threshold
            if last_mouse_x is None:
                last_mouse_x, last_mouse_y = screen_x, screen_y

            smoothed_x = int((1 - SMOOTHING) * last_mouse_x + SMOOTHING * screen_x)
            smoothed_y = int((1 - SMOOTHING) * last_mouse_y + SMOOTHING * screen_y)



            measured = np.array([screen_x, screen_y])
            smooth_x, smooth_y = kf.update(measured)

# --- Velocity dampener and inertia ---
            dx = smooth_x - cursor_x
            dy = smooth_y - cursor_y

            velocity_x = (velocity_x * friction) + (dx * smooth_factor)
            velocity_y = (velocity_y * friction) + (dy * smooth_factor)

            cursor_x += velocity_x
            cursor_y += velocity_y

# --- Keep inside safe bounds ---
            screen_width, screen_height = pyautogui.size()
            safe_x = max(5, min(screen_width - 5, cursor_x))
            safe_y = max(5, min(screen_height - 5, cursor_y))

            pyautogui.moveTo(safe_x, safe_y)


            # --- Scroll ---
            scroll_dir = detect_scroll(lm, base_scale)
            if scroll_dir == "up" and (now - scroll_time) > SCROLL_INTERVAL:
                pyautogui.press("pageup")
                scroll_time = now
                gesture = "Page Up"
            elif scroll_dir == "down" and (now - scroll_time) > SCROLL_INTERVAL:
                pyautogui.press("pagedown")
                scroll_time = now
                gesture = "Page Down"

            # --- Click + Drag (fist hold) ---
            if is_full_fist(lm, base_scale):
                if not clicking and (now - click_cooldown) > CLICK_HOLD_DELAY:
                    pyautogui.mouseDown()
                    clicking = True
                    gesture = "Clicking (drag)"
            elif is_fist_released(lm, base_scale):
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False
                    click_cooldown = now
                    gesture = "Released"

            # --- Right click ---
            if is_right_click(lm, base_scale) and (now - right_click_time) > RIGHT_CLICK_COOLDOWN:
                pyautogui.click(button="right")
                right_click_time = now
                gesture = "Right Click"

            # --- Three-finger left click ---
            if is_three_finger_click(lm, base_scale) and (now - click_cooldown) > CLICK_HOLD_DELAY:
                pyautogui.click(button="left")
                click_cooldown = now
                gesture = "Left Click (3 fingers)"


            # --- ENTER key (edge triggered) ---
            enter_now = is_enter_gesture(lm, base_scale)
            if enter_now and (not prev_enter_state) and (now - enter_press_time) > ENTER_COOLDOWN:
                pyautogui.press("enter")
                enter_press_time = now
                
                gesture = "ENTER ⏎"
            prev_enter_state = enter_now

            # --- Rock & Roll Rickroll (for fun, 10s cooldown) ---
            if rock_and_roll(lm, base_scale):
                if (now - last_rickroll_time) > 10:
                    # Open asynchronously (webbrowser.open is already non-blocking enough here)
                    webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                    last_rickroll_time = now

            # --- Thumbs Up -> set current position as new center ---
            if is_thumbs_up(lm, base_scale):
              custom_center = pyautogui.position()
              center_x_offset = custom_center[0]
              center_y_offset = custom_center[1]
              gesture = "Center Set 👍"
              time.sleep(0.5)  # small delay to prevent multiple triggers

            # --- Shaka (chill) -> reset center to default ---
            if is_shaka(lm, base_scale):
              custom_center = None
              center_x_offset = 0
              center_y_offset = 0
              gesture = "Center Reset 🤙"
              time.sleep(0.5)


        else:
            # If paused, ensure no dangling click
            if clicking:
                pyautogui.mouseUp()
                clicking = False

    else:
        # Lost hand -> reset some transient states
        clicking = False
        prev_enter_state = False
        fist_hold_start = None

    # -------- HUD / Diagnostics --------
    t_now = time.perf_counter()
    dt = t_now - t_last
    if dt > 0:
        fps = 1.0 / dt
    t_last = t_now

    hud1 = f"Gesture: {gesture}   |   FPS: {fps:.1f}"
    hud2 = f"Active: {'YES' if GESTURE_ACTIVE else 'NO (hold fist to toggle)'}"
    color = (0, 255, 0) if GESTURE_ACTIVE else (0, 0, 255)
    cv2.putText(small, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(small, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Touchless Interface ALPHA (Optimized)", small)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

# -------- Cleanup --------
cap.release()
cv2.destroyAllWindows()