import cv2
import sys
import math

# Configure UTF-8 output to prevent Windows console encoding crashes with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass
import warnings
import os
import time
import ctypes

# Ensure relative paths resolve correctly when run via shortcut/background processes
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Use winsound for Windows (built-in, zero lag for wav files)
try:
    import winsound
    USE_WINSOUND = True
except ImportError:
    from playsound import playsound
    USE_WINSOUND = False

# Suppress harmless TensorFlow/MediaPipe logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import mediapipe as mp
from pynput.mouse import Controller, Button
from pynput import keyboard as kb
import tkinter as tk

# =========================================================
# CONFIGURATION SETTINGS
# =========================================================
SHOW_UI = False         # Set to True to see the camera feed
MOUSE_SPEED = 5.5       # Amplifies finger movement distance

# ── Floating Status Dot UI ───────────────────────────────
ENABLE_DOT_UI = False         # Set to True to show the floating status dot on screen
DOT_SIZE = 16                # Diameter of the dot in pixels
# Position: "bottom-right", "bottom-left", "top-right", "top-left", or custom (x, y) tuple e.g. (1800, 1000)
DOT_POSITION = "bottom-right"
DOT_MARGIN_X = 40            # Margin from left/right edge of screen (in pixels)
DOT_MARGIN_Y = 40            # Margin from top/bottom edge of screen (in pixels)

# Status Colors (Hex format or Tkinter color names)
# Format: (Center color, Outer border color)
DOT_COLOR_HAND_DETECTED = ("#00FF00", "#A5D6A7")  # Green center, light green outer ring
DOT_COLOR_MEDIA_NAV     = ("#FFFF00", "#FFF59D")  # Yellow center, light yellow outer ring
DOT_COLOR_NO_HAND       = ("#FF0000", "#EF9A9A")  # Red center, light pink/red outer ring

# ── Dynamic Smoothing (Velocity-Adaptive) ────────────────
# At slow hand speed → SMOOTH_SLOW for pixel-perfect precision
# At fast hand speed → SMOOTH_FAST for zero-lag snapping
SMOOTH_SLOW = 6        # Higher = smoother at low speed
SMOOTH_FAST = 2      # Lower = snappier at high speed
VELOCITY_THRESHOLD = 16   # px/frame speed that fully triggers fast mode

# ── Dynamic Dead-Zone ────────────────────────────────────
DEADZONE = 0.8           # Ignores sub-pixel tremors (px)

# ── Gesture State Machine: Required Confidence Frames ────
# Every gesture needs N consecutive frames before it fires.
GESTURE_CONFIRM_FRAMES = 3

# ── Hand-Size Normalized Thresholds (RATIOS) ─────────────
# All thresholds are ratios of the wrist→middle-knuckle distance.
# This makes them physically constant regardless of camera distance.
LEFT_CLICK_START   = 0.32   # Pinch ratio to start left-click
LEFT_CLICK_RELEASE = 0.46   # Pinch ratio to release left-click

RIGHT_CLICK_START   = 0.32
RIGHT_CLICK_RELEASE = 0.46

SCROLL_START   = 0.32
SCROLL_RELEASE = 0.46

ZOOM_PINCH_THRESH = 0.36    # Per-hand pinch ratio to enter zoom
ZOOM_MIN_SPREAD   = 0.08    # Normalized min two-hand spread

SCROLL_SENSITIVITY  = 0.25
VOLUME_SENSITIVITY  = 0.44
ZOOM_SENSITIVITY    = 50.0

# ── Open Palm Rotation — Media Controls ───────────────────
# All 5 fingers extended + wrist rotation fires next/prev track.
#   Rotate right (clockwise)        → Next Track ⏭️
#   Rotate left  (counter-clockwise) → Prev Track ⏮️
ROTATION_THRESH       = 0.34   # radians (~26°) from baseline to trigger
MEDIA_COOLDOWN_FRAMES = 60     # 3 s at ~30 fps cooldown between commands

# ── Navigation Mode (Fist ✊ gesture) ────────────────────
# Normalized x-distance of fist movement per alt+tab step.
NAV_TAB_SENSITIVITY = 0.05   # in [0..1] units (normalized hand x)
NAV_TAB_COOLDOWN    = 7      # min frames between tab presses (debounce)

# Audio Assets
LEFT_CLICK_SOUND  = "left_click.wav"
RIGHT_CLICK_SOUND = "right_click.wav"
ON_SOUND  = "on.wav"
OFF_SOUND = "off.wav"

# ── Performance Settings ─────────────────────────────────
TARGET_FPS       = 25          # Cap the main loop to this many frames per second
FRAME_INTERVAL   = 1.0 / TARGET_FPS   # seconds per frame (~67 ms)
# Reduced resolution: 320x240 is 1/4 the pixels of 640x480, dramatically
# cutting MediaPipe's per-frame cost. Gesture ratios are normalized so
# accuracy is unaffected.
FRAME_W, FRAME_H = 320, 240
# Tkinter dot UI update throttle: update every N main-loop frames (saves CPU)
DOT_UPDATE_INTERVAL = 3
# Idle frame-skip: when no hand is visible, process only 1 in N frames
IDLE_SKIP_FRAMES = 3
# =========================================================

print("🚀 Initializing Multi-Finger Gesture Engine v2 (Earthquake-Proof Edition)...")

def play_sound(sound_file):
    if os.path.exists(sound_file):
        if USE_WINSOUND:
            winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            playsound(sound_file, block=False)
    else:
        print(f"⚠️ Audio File Missing: {sound_file}")

def send_volume_up():
    try:
        ctypes.windll.user32.keybd_event(0xAF, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0xAF, 0, 2, 0)
    except Exception:
        pass

def send_volume_down():
    try:
        ctypes.windll.user32.keybd_event(0xAE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0xAE, 0, 2, 0)
    except Exception:
        pass

def send_next_track():
    try:
        ctypes.windll.user32.keybd_event(0xB0, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0xB0, 0, 2, 0)
    except Exception:
        pass

def send_prev_track():
    try:
        ctypes.windll.user32.keybd_event(0xB1, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0xB1, 0, 2, 0)
    except Exception:
        pass

def send_play_pause():
    try:
        ctypes.windll.user32.keybd_event(0xB3, 0, 0, 0)
        ctypes.windll.user32.keybd_event(0xB3, 0, 2, 0)
    except Exception:
        pass


# =========================================================
# FLOATING STATUS DOT UI (Transparent & Windowless)
# =========================================================
class FloatingDotUI:
    def __init__(self):
        if not ENABLE_DOT_UI:
            return
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", "black")
        self.root.config(bg="black")
        
        # Calculate position
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        if isinstance(DOT_POSITION, (tuple, list)) and len(DOT_POSITION) == 2:
            x, y = DOT_POSITION
        elif DOT_POSITION == "bottom-right":
            x = screen_w - DOT_SIZE - DOT_MARGIN_X
            y = screen_h - DOT_SIZE - DOT_MARGIN_Y
        elif DOT_POSITION == "bottom-left":
            x = DOT_MARGIN_X
            y = screen_h - DOT_SIZE - DOT_MARGIN_Y
        elif DOT_POSITION == "top-right":
            x = screen_w - DOT_SIZE - DOT_MARGIN_X
            y = DOT_MARGIN_Y
        elif DOT_POSITION == "top-left":
            x = DOT_MARGIN_X
            y = DOT_MARGIN_Y
        else:
            x = screen_w - DOT_SIZE - DOT_MARGIN_X
            y = screen_h - DOT_SIZE - DOT_MARGIN_Y
            
        self.root.geometry(f"{DOT_SIZE}x{DOT_SIZE}+{int(x)}+{int(y)}")
        
        self.canvas = tk.Canvas(self.root, width=DOT_SIZE, height=DOT_SIZE, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack()
        
        self.outer_id = self.canvas.create_oval(1, 1, DOT_SIZE - 1, DOT_SIZE - 1, fill=DOT_COLOR_NO_HAND[1], outline="")
        
        # Inner circle (centered, around 60% of dot size)
        inner_r = max(4, int(DOT_SIZE * 0.6))
        offset = (DOT_SIZE - inner_r) // 2
        self.inner_id = self.canvas.create_oval(offset, offset, offset + inner_r, offset + inner_r, fill=DOT_COLOR_NO_HAND[0], outline="")
        self.current_color = DOT_COLOR_NO_HAND
        
        self.root.update()

    def set_status(self, status):
        """status: 'no_hand', 'hand_detected', 'media_nav'"""
        if not ENABLE_DOT_UI or not hasattr(self, 'root'):
            return
        if status == 'no_hand':
            colors = DOT_COLOR_NO_HAND
        elif status == 'media_nav':
            colors = DOT_COLOR_MEDIA_NAV
        else:
            colors = DOT_COLOR_HAND_DETECTED
            
        if colors != self.current_color:
            self.canvas.itemconfig(self.outer_id, fill=colors[1])
            self.canvas.itemconfig(self.inner_id, fill=colors[0])
            self.current_color = colors
            
    def update(self):
        if not ENABLE_DOT_UI or not hasattr(self, 'root'):
            return
        try:
            self.root.update()
        except Exception:
            pass
            
    def close(self):
        if not ENABLE_DOT_UI or not hasattr(self, 'root'):
            return
        try:
            self.root.destroy()
        except Exception:
            pass


# Initialize Mouse
mouse = Controller()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,           # 0 = lite model (~40% cheaper than default)
    max_num_hands=2,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.80
)

# Camera Setup — 320×240 (¼ the pixels of 640×480, cuts MediaPipe cost dramatically)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)   # Ask driver to cap frame delivery
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)            # Minimize capture buffer lag

# ─── Tracking & State Variables ─────────────────────────
left_clicked  = False
right_clicked = False
scrolling     = False

smooth_x, smooth_y = None, None
prev_smooth_x, prev_smooth_y = None, None  # for velocity calculation
prev_scroll_y = None
was_hand_detected = False

# Zoom state
prev_zoom_spread = None
zoom_accumulator = 0.0
kbd = kb.Controller()

# Volume state
volume_active      = False
prev_volume_y      = None
volume_accumulator = 0.0

# ── Open Palm Rotation state ────────────────────────────
rotation_base_angle = None    # hand angle when open-palm was first detected
media_cooldown      = 0       # frames remaining before next track command fires

# ── Navigation Mode state (Fist ✊) ─────────────────────
nav_mode_active  = False      # True while fist is held
nav_alt_held     = False      # True while Alt key is being held for alt+tab
nav_tab_accum    = 0.0        # accumulated normalized x-movement
prev_nav_x       = None       # previous fist x position
nav_tab_cooldown = 0          # frames until next tab press is allowed

# ─── Universal Gesture State Machine ────────────────────
# Counter goes +1 when gesture detected raw, -1 when not.
# Gesture only "fires" once the counter reaches GESTURE_CONFIRM_FRAMES.
gesture_counters = {
    "peace":    0,
    "fist":     0,   # ✊ fist + thumb out → Navigation Mode
    "pinch_l":  0,   # left-click  (thumb + index)
    "pinch_r":  0,   # right-click (thumb + ring)
    "pinch_m":  0,   # scroll      (thumb + middle)
}
gesture_states = {k: False for k in gesture_counters}

def update_gesture(name, raw):
    """Feed a raw boolean into the state machine. Returns stable confirmed state."""
    N = GESTURE_CONFIRM_FRAMES
    if raw:
        gesture_counters[name] = min(N, gesture_counters[name] + 1)
    else:
        gesture_counters[name] = max(0, gesture_counters[name] - 1)
    gesture_states[name] = (gesture_counters[name] >= N)
    return gesture_states[name]

def reset_all_gestures():
    for k in gesture_counters:
        gesture_counters[k] = 0
        gesture_states[k]   = False

# ─── Utility Functions ───────────────────────────────────
def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_hand_scale(lm):
    """
    UPGRADE 1 – Dynamic Hand-Size Normalization.
    Distance from wrist (0) to middle-finger MCP (9).
    Shrinks/grows proportionally with camera distance,
    so normalized ratios stay physically constant.
    """
    return get_distance(lm[0], lm[9])

def get_pinch_midpoint(lm):
    """Midpoint between thumb tip (4) and index tip (8)."""
    return (lm[4].x + lm[8].x) / 2.0, (lm[4].y + lm[8].y) / 2.0

def is_finger_up(lm, tip_id, pip_id):
    """True if fingertip is further from wrist than its PIP joint."""
    return get_distance(lm[0], lm[tip_id]) > get_distance(lm[0], lm[pip_id])

dot_ui = None
print("\n🚀 Camera Online. System tracking live. Press 'q' to quit.")

# ─── Performance counters ────────────────────────────────
_frame_count    = 0      # total frames processed
_idle_counter   = 0      # counts frames while no hand, used for idle skip
_dot_counter    = 0      # counts frames for Tkinter throttle
_last_frame_t   = None   # wall-clock time of the last loop start

try:
    dot_ui = FloatingDotUI()
    while cap.isOpened():
        # ── FPS CAP: sleep for the remainder of the frame interval ──────
        now = time.monotonic()
        if _last_frame_t is not None:
            elapsed = now - _last_frame_t
            remaining = FRAME_INTERVAL - elapsed
            if remaining > 0.001:
                time.sleep(remaining)
        _last_frame_t = time.monotonic()

        success, frame = cap.read()
        if not success:
            break

        # ── IDLE FRAME SKIP: when no hand, process only every Nth frame ─
        if not was_hand_detected:
            _idle_counter += 1
            if _idle_counter % IDLE_SKIP_FRAMES != 0:
                continue   # skip full MediaPipe processing this frame

        # ── Always flip horizontally ────────────────────────────────────
        # IMPORTANT: flip must happen BEFORE MediaPipe so landmark x-coords
        # match screen direction. Without flip, moving hand right makes
        # cursor go left (raw camera x is mirrored vs screen x).
        frame = cv2.flip(frame, 1)

        # ── OPTIMIZATION: mark frame non-writeable → MediaPipe skips copy ─
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame.flags.writeable = True   # restore for drawing if SHOW_UI

        current_hand_detected = bool(results.multi_hand_landmarks)

        # --- STATE TRANSITION: HAND DETECTED (OFF → ON) ---
        if current_hand_detected and not was_hand_detected:
            print("👋 Hand Detected! Activating in 0.5s...")
            play_sound(ON_SOUND)
            time.sleep(0.05)
            smooth_x, smooth_y = None, None
            prev_smooth_x, prev_smooth_y = None, None
            prev_scroll_y = None
            prev_volume_y = None
            reset_all_gestures()

        # --- STATE TRANSITION: HAND LOST (ON → OFF) ---
        if not current_hand_detected and was_hand_detected:
            print("📴 Hand Lost.")
            play_sound(OFF_SOUND)
            smooth_x, smooth_y = None, None
            prev_smooth_x, prev_smooth_y = None, None
            prev_scroll_y = None
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False
            if right_clicked:
                mouse.release(Button.right)
                right_clicked = False
            scrolling          = False
            volume_active      = False
            prev_volume_y      = None
            volume_accumulator = 0.0
            rotation_base_angle = None
            media_cooldown      = 0
            if nav_mode_active and nav_alt_held:
                try:
                    kbd.release(kb.Key.alt)
                except Exception:
                    pass
            nav_mode_active    = False
            nav_alt_held       = False
            nav_tab_accum      = 0.0
            prev_nav_x         = None
            nav_tab_cooldown   = 0
            reset_all_gestures()

        was_hand_detected = current_hand_detected

        # --- CORE VISION PROCESSING ---
        if current_hand_detected:
            n_hands = len(results.multi_hand_landmarks)

            # ── TWO-HAND ZOOM ──────────────────────────────────────────
            if n_hands == 2:
                lm0 = results.multi_hand_landmarks[0].landmark
                lm1 = results.multi_hand_landmarks[1].landmark

                scale0 = get_hand_scale(lm0)
                scale1 = get_hand_scale(lm1)

                # Normalize each hand's pinch by its own live scale
                pinch0 = (get_distance(lm0[4], lm0[8]) / scale0 < ZOOM_PINCH_THRESH) if scale0 > 0 else False
                pinch1 = (get_distance(lm1[4], lm1[8]) / scale1 < ZOOM_PINCH_THRESH) if scale1 > 0 else False

                if pinch0 and pinch1:
                    mx0, my0 = get_pinch_midpoint(lm0)
                    mx1, my1 = get_pinch_midpoint(lm1)
                    spread = math.sqrt((mx0 - mx1)**2 + (my0 - my1)**2)

                    if spread > ZOOM_MIN_SPREAD:
                        if prev_zoom_spread is not None:
                            alpha = 0.15
                            smoothed_spread = alpha * spread + (1.0 - alpha) * prev_zoom_spread
                            delta = smoothed_spread - prev_zoom_spread
                            if abs(delta) < 0.002:
                                delta = 0.0
                            zoom_accumulator += delta * ZOOM_SENSITIVITY
                            clicks = int(zoom_accumulator)
                            if clicks != 0:
                                zoom_accumulator -= clicks
                                try:
                                    kbd.press(kb.Key.ctrl)
                                    mouse.scroll(0, clicks)
                                    kbd.release(kb.Key.ctrl)
                                except Exception:
                                    pass
                            prev_zoom_spread = smoothed_spread
                        else:
                            prev_zoom_spread = spread
                    else:
                        prev_zoom_spread = None
                        zoom_accumulator = 0.0

                    if SHOW_UI:
                        for hl in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(frame, f"ZOOM spread={spread:.3f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                    continue   # skip single-hand logic while both pinching
                else:
                    prev_zoom_spread = None
                    zoom_accumulator = 0.0
            else:
                prev_zoom_spread = None
                zoom_accumulator = 0.0

            # ── SINGLE HAND PROCESSING ─────────────────────────────────
            for hand_landmarks in results.multi_hand_landmarks:
                if SHOW_UI:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark

                # ── UPGRADE 1: Live hand scale ────────────────────────
                hand_scale = get_hand_scale(lm)
                if hand_scale < 0.001:
                    continue  # degenerate landmark set; skip this frame

                # Normalized pinch ratios (physically constant at any distance)
                left_ratio   = get_distance(lm[4], lm[8])  / hand_scale
                scroll_ratio = get_distance(lm[4], lm[12]) / hand_scale
                right_ratio  = get_distance(lm[4], lm[16]) / hand_scale

                # Cursor anchor: index MCP knuckle (landmark 5)
                cx = lm[5].x * FRAME_W
                cy = lm[5].y * FRAME_H

                # ── Finger-up classification ──────────────────────────
                index_up  = is_finger_up(lm, 8,  6)
                middle_up = is_finger_up(lm, 12, 10)
                ring_up   = is_finger_up(lm, 16, 14)
                pinky_up  = is_finger_up(lm, 20, 18)

                # ── UPGRADE 2: Unified Gesture State Machine ──────────
                any_pinching = (
                    left_ratio   < LEFT_CLICK_START or
                    scroll_ratio < SCROLL_START     or
                    right_ratio  < RIGHT_CLICK_START
                )

                # Thumb-up check: tip further from wrist than IP joint
                thumb_up = get_distance(lm[0], lm[4]) > get_distance(lm[0], lm[3])

                # Raw (un-buffered) gesture detections
                raw_peace = index_up and middle_up and not ring_up and not pinky_up and not any_pinching
                # ✊ fist-with-thumb-out: 4 fingers curled, thumb extended
                raw_fist  = (thumb_up and not index_up and not middle_up
                             and not ring_up and not pinky_up and not any_pinching)
                # 🖐️ open palm: all 5 fingers extended (used for rotation media control)
                all_fingers_up = (thumb_up and index_up and middle_up
                                  and ring_up and pinky_up and not any_pinching)
                raw_pinch_l = left_ratio   < LEFT_CLICK_START
                raw_pinch_r = right_ratio  < RIGHT_CLICK_START
                raw_pinch_m = scroll_ratio < SCROLL_START

                # Feed through state machine (each needs N consecutive frames)
                is_peace   = update_gesture("peace",   raw_peace)
                is_fist    = update_gesture("fist",    raw_fist)
                is_pinch_l = update_gesture("pinch_l", raw_pinch_l)
                is_pinch_r = update_gesture("pinch_r", raw_pinch_r)
                is_pinch_m = update_gesture("pinch_m", raw_pinch_m)

                # Hysteresis release thresholds (raw, no buffer needed for release)
                pinch_l_released = left_ratio   > LEFT_CLICK_RELEASE
                pinch_r_released = right_ratio  > RIGHT_CLICK_RELEASE
                pinch_m_released = scroll_ratio > SCROLL_RELEASE

                # ── MEDIA CONTROLS: OPEN PALM WRIST ROTATION 🖐️ ──────────
                # All 5 fingers extended. Wrist rotates:
                #   Clockwise (right)         → Next Track ⏭️   (3 s cooldown)
                #   Counter-clockwise (left)  → Prev Track ⏮️   (3 s cooldown)
                # Angle measured from wrist (lm[0]) to middle MCP (lm[9]).
                if all_fingers_up and media_cooldown <= 0:
                    cur_angle = math.atan2(
                        lm[9].y - lm[0].y,
                        lm[9].x - lm[0].x
                    )
                    if rotation_base_angle is None:
                        rotation_base_angle = cur_angle
                    else:
                        delta = cur_angle - rotation_base_angle
                        # Normalise to (−π, +π] to handle wrap-around
                        if delta > math.pi:
                            delta -= 2 * math.pi
                        elif delta < -math.pi:
                            delta += 2 * math.pi

                        if delta > ROTATION_THRESH:
                            send_next_track()
                            print("⏭️  Next Track (rotate right)")
                            media_cooldown      = MEDIA_COOLDOWN_FRAMES
                            rotation_base_angle = None
                        elif delta < -ROTATION_THRESH:
                            send_prev_track()
                            print("⏮️  Prev Track (rotate left)")
                            media_cooldown      = MEDIA_COOLDOWN_FRAMES
                            rotation_base_angle = None
                else:
                    rotation_base_angle = None  # reset baseline when hand leaves pose

                if media_cooldown > 0:
                    media_cooldown -= 1

                # ── NAVIGATION MODE: FIST ✊ ─────────────────────────
                # Entry: confirmed fist (thumb + all fingers curled, no pinching).
                # While active:
                #   Move fist RIGHT → Tab       (next app in Alt+Tab)
                #   Move fist LEFT  → Shift+Tab (prev app in Alt+Tab)
                # Exit (open fist) → release Alt → selected window activates.
                # Cursor movement and clicks are DISABLED in nav mode.
                if is_fist:
                    if not nav_mode_active:
                        nav_mode_active  = True
                        nav_alt_held     = True
                        nav_tab_accum    = 0.0
                        nav_tab_cooldown = 0
                        prev_nav_x       = lm[5].x
                        try:
                            kbd.press(kb.Key.alt)
                            kbd.press(kb.Key.tab)
                            kbd.release(kb.Key.tab)
                        except Exception:
                            pass
                        print("✊ Navigation Mode ON")
                    else:
                        cur_nav_x = lm[5].x
                        if prev_nav_x is not None:
                            dx_nav = cur_nav_x - prev_nav_x
                            nav_tab_accum += dx_nav

                            if nav_tab_cooldown <= 0:
                                if nav_tab_accum > NAV_TAB_SENSITIVITY:
                                    try:
                                        kbd.press(kb.Key.tab)
                                        kbd.release(kb.Key.tab)
                                    except Exception:
                                        pass
                                    nav_tab_accum    = 0.0
                                    nav_tab_cooldown = NAV_TAB_COOLDOWN
                                    print("→ Alt+Tab: Next")
                                elif nav_tab_accum < -NAV_TAB_SENSITIVITY:
                                    try:
                                        kbd.press(kb.Key.shift)
                                        kbd.press(kb.Key.tab)
                                        kbd.release(kb.Key.tab)
                                        kbd.release(kb.Key.shift)
                                    except Exception:
                                        pass
                                    nav_tab_accum    = 0.0
                                    nav_tab_cooldown = NAV_TAB_COOLDOWN
                                    print("← Alt+Tab: Prev")

                        if nav_tab_cooldown > 0:
                            nav_tab_cooldown -= 1
                        prev_nav_x = cur_nav_x
                else:
                    if nav_mode_active:
                        nav_mode_active = False
                        nav_tab_accum   = 0.0
                        prev_nav_x      = None
                        if nav_alt_held:
                            try:
                                kbd.release(kb.Key.alt)
                            except Exception:
                                pass
                            nav_alt_held = False
                        print("✊ Navigation Mode OFF — window selected")


                # ── VOLUME CONTROL: PEACE SIGN ────────────────────────
                if is_peace:
                    if not volume_active:
                        volume_active = True
                        prev_volume_y = cy
                        volume_accumulator = 0.0
                    else:
                        if prev_volume_y is not None:
                            dy_vol = cy - prev_volume_y
                            volume_accumulator += -dy_vol * VOLUME_SENSITIVITY
                            steps = int(volume_accumulator / 10.0)
                            if steps != 0:
                                volume_accumulator -= steps * 10.0
                                for _ in range(abs(steps)):
                                    if steps > 0:
                                        send_volume_up()
                                    else:
                                        send_volume_down()
                                prev_volume_y = cy
                else:
                    if volume_active:
                        volume_active = False
                        prev_volume_y = None
                        volume_accumulator = 0.0

                # ── SCROLL MODE: THUMB + MIDDLE PINCH ────────────────
                if is_pinch_m and not volume_active:
                    if not scrolling:
                        scrolling = True
                        prev_scroll_y = cy
                    else:
                        if prev_scroll_y is not None:
                            dy_scroll = cy - prev_scroll_y
                            scroll_amount = int(-dy_scroll * SCROLL_SENSITIVITY)
                            if scroll_amount != 0:
                                mouse.scroll(0, scroll_amount)
                                prev_scroll_y = cy
                elif pinch_m_released or volume_active:
                    if scrolling:
                        scrolling = False
                        prev_scroll_y = None

                # ── UPGRADE 3: VELOCITY-ADAPTIVE CURSOR MOVEMENT ──────
                # Cursor moves only when right-click is open, not scrolling,
                # and not in volume mode.
                if pinch_r_released and not scrolling and not volume_active and not nav_mode_active:
                    if smooth_x is None or smooth_y is None:
                        smooth_x, smooth_y = cx, cy
                        prev_smooth_x, prev_smooth_y = cx, cy
                    else:
                        # Calculate raw hand velocity in screen pixels/frame
                        if prev_smooth_x is not None:
                            vel = math.sqrt(
                                (cx - prev_smooth_x)**2 +
                                (cy - prev_smooth_y)**2
                            )
                        else:
                            vel = 0.0

                        # Linearly interpolate smoothing factor:
                        # Slow hand → max smoothing (sniper precision)
                        # Fast hand → min smoothing (instant snap)
                        t = min(1.0, vel / VELOCITY_THRESHOLD)
                        dynamic_smooth = SMOOTH_SLOW + t * (SMOOTH_FAST - SMOOTH_SLOW)

                        next_smooth_x = smooth_x + (cx - smooth_x) / dynamic_smooth
                        next_smooth_y = smooth_y + (cy - smooth_y) / dynamic_smooth

                        dx = (next_smooth_x - smooth_x) * MOUSE_SPEED
                        dy = (next_smooth_y - smooth_y) * MOUSE_SPEED

                        if abs(dx) > DEADZONE or abs(dy) > DEADZONE:
                            curr_mx, curr_my = mouse.position
                            mouse.position = (int(curr_mx + dx), int(curr_my + dy))

                        prev_smooth_x, prev_smooth_y = smooth_x, smooth_y
                        smooth_x, smooth_y = next_smooth_x, next_smooth_y
                else:
                    smooth_x, smooth_y = None, None
                    prev_smooth_x, prev_smooth_y = None, None

                # ── LEFT CLICK / DRAG ─────────────────────────────────
                if is_pinch_l and not volume_active and not nav_mode_active:
                    if not left_clicked:
                        mouse.press(Button.left)
                        play_sound(LEFT_CLICK_SOUND)
                        left_clicked = True
                elif pinch_l_released or volume_active or nav_mode_active:
                    if left_clicked:
                        mouse.release(Button.left)
                        left_clicked = False

                # ── RIGHT CLICK (THUMB + RING) ────────────────────────
                if is_pinch_r and not volume_active and not nav_mode_active:
                    if not right_clicked:
                        mouse.press(Button.right)
                        play_sound(RIGHT_CLICK_SOUND)
                        right_clicked = True
                elif pinch_r_released or volume_active or nav_mode_active:
                    if right_clicked:
                        mouse.release(Button.right)
                        right_clicked = False

                # ── UI DEBUG OVERLAY ──────────────────────────────────
                if SHOW_UI:
                    mode = "CURSOR"
                    if nav_mode_active:   mode = "NAV"
                    elif media_cooldown > 0: mode = "MEDIA-CD"
                    elif volume_active:   mode = "VOLUME"
                    elif scrolling:       mode = "SCROLL"
                    elif left_clicked:    mode = "L-DRAG"
                    elif right_clicked:   mode = "R-CLICK"
                    cv2.putText(frame, f"MODE: {mode}  scale={hand_scale:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 120), 2)
                    cv2.putText(frame, f"L={left_ratio:.2f} R={right_ratio:.2f} M={scroll_ratio:.2f}", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        # ── UPDATE FLOATING STATUS DOT (throttled) ────────────
        _dot_counter += 1
        if dot_ui:
            # Always keep status current; only call update() every N frames
            if current_hand_detected:
                if nav_mode_active or volume_active or (media_cooldown > 0):
                    dot_ui.set_status('media_nav')
                else:
                    dot_ui.set_status('hand_detected')
            else:
                dot_ui.set_status('no_hand')
            if _dot_counter % DOT_UPDATE_INTERVAL == 0:
                dot_ui.update()

        if SHOW_UI:
            if nav_mode_active:
                cv2.putText(frame, "NAV MODE  fist right=Next  left=Prev  open fist=Confirm", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            elif media_cooldown > 0:
                cv2.putText(frame, f"MEDIA cooldown: {media_cooldown // 30 + 1}s", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
            elif volume_active:
                cv2.putText(frame, "VOLUME ACTIVE", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow('Gesture Mouse v2 — Debug Panel', frame)
            # waitKey handles display refresh; remaining sleep already done above
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nProcess interrupted cleanly.")

finally:
    if dot_ui:
        dot_ui.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 System offline.")