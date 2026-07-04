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
SMOOTHENING = 5.0      # Lower value = faster response, Higher value = smoother
MOUSE_SPEED = 5.0      # Amplifies finger movement distance
DEADZONE = 1.2         # Ignores movements smaller than this many pixels

# User Calibrated Thresholds (Dual-Threshold Hysteresis Buffer)
LEFT_CLICK_START = 0.04   
LEFT_CLICK_RELEASE = 0.06 

RIGHT_CLICK_START = 0.04
RIGHT_CLICK_RELEASE = 0.06

SCROLL_START = 0.04
SCROLL_RELEASE = 0.06
SCROLL_SENSITIVITY = 0.25  # Adjust this to make scrolling faster or slower
VOLUME_SENSITIVITY = 0.44  # Adjust this to make volume adjustment faster or slower

# Zoom (two-hand) thresholds
ZOOM_PINCH_THRESH = 0.1   # max dist for "is pinching"
ZOOM_SENSITIVITY = 50.0    # px-of-spread per scroll click
ZOOM_MIN_SPREAD = 0.1      # ignore spreads smaller than this

# Audio Assets
LEFT_CLICK_SOUND = "left_click2.wav"
RIGHT_CLICK_SOUND = "right_click.wav"
ON_SOUND = "on.wav"
OFF_SOUND = "off.wav"
# =========================================================

print("🚀 Initializing Multi-Finger Gesture Engine...")

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

# Tracking and State management variables
left_clicked = False
right_clicked = False
scrolling = False

smooth_x, smooth_y = None, None
prev_scroll_y = None
was_hand_detected = False  

# Zoom state
prev_zoom_spread = None
zoom_accumulator = 0.0
kbd = kb.Controller()

# Volume state
volume_active = False
prev_volume_y = None
volume_accumulator = 0.0
peace_counter = 0

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_pinch_midpoint(lm):
    """Midpoint between thumb tip (4) and index tip (8)."""
    return (lm[4].x + lm[8].x) / 2.0, (lm[4].y + lm[8].y) / 2.0

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
        
        # --- STATE TRANSITION: HAND DETECTED (OFF -> ON) ---
        if current_hand_detected and not was_hand_detected:
            print("👋 Hand Detected! Activating in 0.5s...")
            play_sound(ON_SOUND)
            time.sleep(0.5)  # User Tweak: 0.5-second snappier pause
            smooth_x, smooth_y = None, None  
            prev_scroll_y = None
            prev_volume_y = None
            peace_counter = 0
            
        # --- STATE TRANSITION: HAND LOST (ON -> OFF) ---
        if not current_hand_detected and was_hand_detected:
            print("📴 Hand Lost.")
            play_sound(OFF_SOUND)
            smooth_x, smooth_y = None, None
            prev_scroll_y = None
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False
            if right_clicked:
                mouse.release(Button.right)
                right_clicked = False
            scrolling = False
            volume_active = False
            prev_volume_y = None
            volume_accumulator = 0.0
            peace_counter = 0

        was_hand_detected = current_hand_detected
        
        # --- CORE VISION PROCESSING ---
        if current_hand_detected:
            n_hands = len(results.multi_hand_landmarks)

            # ── TWO-HAND ZOOM ─────────────────────────────────────────────────
            if n_hands == 2:
                lm0 = results.multi_hand_landmarks[0].landmark
                lm1 = results.multi_hand_landmarks[1].landmark
                pinch0 = get_distance(lm0[4], lm0[8]) < ZOOM_PINCH_THRESH
                pinch1 = get_distance(lm1[4], lm1[8]) < ZOOM_PINCH_THRESH

                if pinch0 and pinch1:
                    mx0, my0 = get_pinch_midpoint(lm0)
                    mx1, my1 = get_pinch_midpoint(lm1)
                    spread = math.sqrt((mx0 - mx1)**2 + (my0 - my1)**2)

                    if spread > ZOOM_MIN_SPREAD:
                        if prev_zoom_spread is not None:
                            # 1. EMA Smoothing to reduce jitter
                            alpha = 0.15
                            smoothed_spread = alpha * spread + (1.0 - alpha) * prev_zoom_spread
                            
                            # 2. Delta calculation
                            delta = smoothed_spread - prev_zoom_spread
                            
                            # 3. Dead-zone (ignore sub-pixel tremors)
                            if abs(delta) < 0.002:
                                delta = 0.0
                                
                            # 4. Less sensitive accumulation (no * 10 multiplier)
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

            for hand_landmarks in results.multi_hand_landmarks:
                if SHOW_UI:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                landmarks = hand_landmarks.landmark
                
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]   # Middle finger -> Controls Scroll Mode
                ring_tip = landmarks[16]     # User Tweak: Ring finger -> Controls Right Click
                index_knuckle = landmarks[5] # Stable anchor point for motion
                
                left_dist = get_distance(thumb_tip, index_tip)
                scroll_dist = get_distance(thumb_tip, middle_tip)
                right_dist = get_distance(thumb_tip, ring_tip)
                
                cx, cy = index_knuckle.x * FRAME_W, index_knuckle.y * FRAME_H
                
                # --- GESTURE CLASSIFIER / PEACE SIGN DETECTION ---
                index_up = get_distance(landmarks[0], landmarks[8]) > get_distance(landmarks[0], landmarks[6])
                middle_up = get_distance(landmarks[0], landmarks[12]) > get_distance(landmarks[0], landmarks[10])
                ring_down = get_distance(landmarks[0], landmarks[16]) < get_distance(landmarks[0], landmarks[14])
                pinky_down = get_distance(landmarks[0], landmarks[20]) < get_distance(landmarks[0], landmarks[18])
                
                is_pinching = (left_dist < LEFT_CLICK_START or scroll_dist < SCROLL_START or right_dist < RIGHT_CLICK_START)
                is_peace_raw = index_up and middle_up and ring_down and pinky_down and not is_pinching
                
                if is_peace_raw:
                    peace_counter = min(5, peace_counter + 1)
                else:
                    peace_counter = max(0, peace_counter - 1)
                
                is_peace = (peace_counter >= 3)
                
                # --- VOLUME FEATURE LOGIC ---
                if is_peace:
                    if not volume_active:
                        volume_active = True
                        prev_volume_y = cy
                        volume_accumulator = 0.0
                    else:
                        if prev_volume_y is not None:
                            dy = cy - prev_volume_y
                            # Hand moving UP (decreasing cy) increases volume.
                            # Hand moving DOWN (increasing cy) decreases volume.
                            # Make it very less sensitive using the configuration variable.
                            volume_accumulator += -dy * VOLUME_SENSITIVITY
                            
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

                # --- SCROLL FEATURE LOGIC ---
                if scroll_dist < SCROLL_START and not volume_active:
                    if not scrolling:
                        scrolling = True
                        prev_scroll_y = cy
                    else:
                        if prev_scroll_y is not None:
                            dy = cy - prev_scroll_y
                            # Moving hand UP decreases cy in camera frame, which should scroll UP (positive value)
                            # Moving hand DOWN increases cy, which should scroll DOWN (negative value)
                            scroll_amount = int(-dy * SCROLL_SENSITIVITY)
                            if scroll_amount != 0:
                                mouse.scroll(0, scroll_amount)
                                prev_scroll_y = cy # Update anchor point for continuous scrolling
                elif scroll_dist > SCROLL_RELEASE or volume_active:
                    if scrolling:
                        scrolling = False
                        prev_scroll_y = None
 
                # --- CURSOR MOVEMENT LOGIC ---
                # Cursor ONLY moves if right-click is open, scroll mode is inactive, and volume mode is inactive
                if right_dist > RIGHT_CLICK_RELEASE and not scrolling and not volume_active:
                    if smooth_x is None or smooth_y is None:
                        smooth_x, smooth_y = cx, cy
                    else:
                        next_smooth_x = smooth_x + (cx - smooth_x) / SMOOTHENING
                        next_smooth_y = smooth_y + (cy - smooth_y) / SMOOTHENING
                        
                        dx = (next_smooth_x - smooth_x) * MOUSE_SPEED
                        dy = (next_smooth_y - smooth_y) * MOUSE_SPEED
                        
                        if abs(dx) > DEADZONE or abs(dy) > DEADZONE:
                            curr_mouse_x, curr_mouse_y = mouse.position
                            mouse.position = (int(curr_mouse_x + dx), int(curr_mouse_y + dy))
                            smooth_x, smooth_y = next_smooth_x, next_smooth_y
                else:
                    # Freeze the cursor anchor while scrolling, right-clicking, or adjusting volume
                    smooth_x, smooth_y = None, None
                
                # --- LEFT CLICK / DRAG LOGIC ---
                if left_dist < LEFT_CLICK_START and not volume_active:  
                    if not left_clicked:
                        mouse.press(Button.left)
                        play_sound(LEFT_CLICK_SOUND)
                        left_clicked = True
                elif left_dist > LEFT_CLICK_RELEASE or volume_active:
                    if left_clicked:
                        mouse.release(Button.left)
                        left_clicked = False
                        
                # --- RIGHT CLICK LOGIC (ON RING FINGER) ---
                if right_dist < RIGHT_CLICK_START and not volume_active:
                    if not right_clicked:
                        mouse.press(Button.right)
                        play_sound(RIGHT_CLICK_SOUND)
                        right_clicked = True
                elif right_dist > RIGHT_CLICK_RELEASE or volume_active:
                    if right_clicked:
                        mouse.release(Button.right)
                        right_clicked = False
 
        if SHOW_UI:
            if volume_active:
                cv2.putText(frame, "VOLUME ACTIVE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow('Gesture Mouse Testing Panel', frame)
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