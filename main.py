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

# =========================================================
# CONFIGURATION SETTINGS
# =========================================================
SHOW_UI = False         # Set to True to see the camera feed
MOUSE_SPEED = 4.0       # Amplifies finger movement distance

# ── Dynamic Smoothing (Velocity-Adaptive) ────────────────
# At slow hand speed → SMOOTH_SLOW for pixel-perfect precision
# At fast hand speed → SMOOTH_FAST for zero-lag snapping
SMOOTH_SLOW = 6        # Higher = smoother at low speed
SMOOTH_FAST = 3       # Lower = snappier at high speed
VELOCITY_THRESHOLD = 16   # px/frame speed that fully triggers fast mode

# ── Dynamic Dead-Zone ────────────────────────────────────
DEADZONE = 1.2           # Ignores sub-pixel tremors (px)

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

# ── Call-Me Media Mode ───────────────────────────────────
# A finger is considered "flicked" when its tip drops by more than this
# fraction of hand-scale within a single frame (fast downward snap).
MEDIA_FLICK_THRESH = 0.25   # ratio of hand_scale per frame = snappy flick

# Audio Assets
LEFT_CLICK_SOUND  = "left_click2.wav"
RIGHT_CLICK_SOUND = "right_click.wav"
ON_SOUND  = "on.wav"
OFF_SOUND = "off.wav"
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

# Initialize Mouse
mouse = Controller()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

# Camera Setup
FRAME_W, FRAME_H = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

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

# Call-Me Media Mode state
media_mode_active  = False    # True while 🤙 pose is held
prev_thumb_y       = None     # tracks thumb tip Y for flick detection
prev_pinky_y       = None     # tracks pinky tip Y for flick detection
thumb_flick_ready  = True     # False while recovering from a fired thumb flick
pinky_flick_ready  = True     # False while recovering from a fired pinky flick

# ─── Universal Gesture State Machine ────────────────────
# Counter goes +1 when gesture detected raw, -1 when not.
# Gesture only "fires" once the counter reaches GESTURE_CONFIRM_FRAMES.
gesture_counters = {
    "peace":    0,
    "call_me":  0,   # 🤙 thumb + pinky up, rest curled
    "fist":     0,
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

print("\n🚀 Camera Online. System tracking live. Press 'q' to quit.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_hand_detected = bool(results.multi_hand_landmarks)

        # --- STATE TRANSITION: HAND DETECTED (OFF → ON) ---
        if current_hand_detected and not was_hand_detected:
            print("👋 Hand Detected! Activating in 0.5s...")
            play_sound(ON_SOUND)
            time.sleep(0.5)
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
            media_mode_active  = False
            prev_thumb_y       = None
            prev_pinky_y       = None
            thumb_flick_ready  = True
            pinky_flick_ready  = True
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
                raw_peace   = index_up and middle_up and not ring_up and not pinky_up and not any_pinching
                # 🤙 call-me: only thumb + pinky extended, middle three curled
                raw_call_me = (thumb_up and pinky_up
                               and not index_up and not middle_up and not ring_up
                               and not any_pinching)
                raw_fist    = not index_up and not middle_up and not ring_up and not pinky_up
                raw_pinch_l = left_ratio   < LEFT_CLICK_START
                raw_pinch_r = right_ratio  < RIGHT_CLICK_START
                raw_pinch_m = scroll_ratio < SCROLL_START

                # Feed through state machine (each needs N consecutive frames)
                is_peace    = update_gesture("peace",   raw_peace)
                is_call_me  = update_gesture("call_me", raw_call_me)
                _is_fist    = update_gesture("fist",    raw_fist)   # reserved for future use
                is_pinch_l  = update_gesture("pinch_l", raw_pinch_l)
                is_pinch_r  = update_gesture("pinch_r", raw_pinch_r)
                is_pinch_m  = update_gesture("pinch_m", raw_pinch_m)

                # Hysteresis release thresholds (raw, no buffer needed for release)
                pinch_l_released = left_ratio   > LEFT_CLICK_RELEASE
                pinch_r_released = right_ratio  > RIGHT_CLICK_RELEASE
                pinch_m_released = scroll_ratio > SCROLL_RELEASE

                # ── MEDIA MODE: CALL-ME 🤙 GESTURE ───────────────────
                # Entry: hold thumb + pinky up (3 confirmed frames).
                # While in media mode, a sharp downward flick of:
                #   pinky only  → Next Track ⏭️
                #   thumb only  → Prev Track ⏮️
                #   both        → Play / Pause ⏯️
                # Exit: drop the call-me pose.
                if is_call_me:
                    if not media_mode_active:
                        # Just entered media mode
                        media_mode_active = True
                        prev_thumb_y = lm[4].y   # thumb tip
                        prev_pinky_y = lm[20].y  # pinky tip
                        thumb_flick_ready = True
                        pinky_flick_ready = True
                        print("🤙 Media Mode ON")
                    else:
                        # Already in media mode — check for flicks each frame.
                        # A "flick" is a fast downward snap of the tip.
                        # We compare raw tip-Y movement (normalised by hand_scale).
                        cur_thumb_y = lm[4].y
                        cur_pinky_y = lm[20].y

                        if prev_thumb_y is not None and prev_pinky_y is not None:
                            dy_thumb = cur_thumb_y - prev_thumb_y  # +ve = moving down
                            dy_pinky = cur_pinky_y - prev_pinky_y

                            # Normalise by hand_scale so the threshold is distance-invariant
                            norm_thumb = dy_thumb / hand_scale
                            norm_pinky = dy_pinky / hand_scale

                            flick_thumb = norm_thumb > MEDIA_FLICK_THRESH and thumb_flick_ready
                            flick_pinky = norm_pinky > MEDIA_FLICK_THRESH and pinky_flick_ready

                            if flick_thumb and flick_pinky:
                                send_play_pause()
                                print("⏯️ Play / Pause")
                                thumb_flick_ready = False
                                pinky_flick_ready = False
                            elif flick_pinky:
                                send_next_track()
                                print("⏭️ Next Track")
                                pinky_flick_ready = False
                            elif flick_thumb:
                                send_prev_track()
                                print("⏮️ Previous Track")
                                thumb_flick_ready = False

                            # Re-arm once the tip returns close to its resting position
                            if not thumb_flick_ready and norm_thumb < 0.0:
                                thumb_flick_ready = True
                            if not pinky_flick_ready and norm_pinky < 0.0:
                                pinky_flick_ready = True

                        prev_thumb_y = cur_thumb_y
                        prev_pinky_y = cur_pinky_y
                else:
                    if media_mode_active:
                        media_mode_active = False
                        prev_thumb_y  = None
                        prev_pinky_y  = None
                        thumb_flick_ready = True
                        pinky_flick_ready = True
                        print("🤙 Media Mode OFF")

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
                if pinch_r_released and not scrolling and not volume_active:
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
                if is_pinch_l and not volume_active:
                    if not left_clicked:
                        mouse.press(Button.left)
                        play_sound(LEFT_CLICK_SOUND)
                        left_clicked = True
                elif pinch_l_released or volume_active:
                    if left_clicked:
                        mouse.release(Button.left)
                        left_clicked = False

                # ── RIGHT CLICK (THUMB + RING) ────────────────────────
                if is_pinch_r and not volume_active:
                    if not right_clicked:
                        mouse.press(Button.right)
                        play_sound(RIGHT_CLICK_SOUND)
                        right_clicked = True
                elif pinch_r_released or volume_active:
                    if right_clicked:
                        mouse.release(Button.right)
                        right_clicked = False

                # ── UI DEBUG OVERLAY ──────────────────────────────────
                if SHOW_UI:
                    mode = "CURSOR"
                    if media_mode_active: mode = "MEDIA"
                    elif volume_active:   mode = "VOLUME"
                    elif scrolling:       mode = "SCROLL"
                    elif left_clicked:    mode = "L-DRAG"
                    elif right_clicked:   mode = "R-CLICK"
                    cv2.putText(frame, f"MODE: {mode}  scale={hand_scale:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 120), 2)
                    cv2.putText(frame, f"L={left_ratio:.2f} R={right_ratio:.2f} M={scroll_ratio:.2f}", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        if SHOW_UI:
            if media_mode_active:
                cv2.putText(frame, "MEDIA MODE  flick pinky=Next  thumb=Prev  both=Play/Pause", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
            elif volume_active:
                cv2.putText(frame, "VOLUME ACTIVE", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow('Gesture Mouse v2 — Debug Panel', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nProcess interrupted cleanly.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("👋 System offline.")