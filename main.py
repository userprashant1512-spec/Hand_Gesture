
import tkinter as tk
from tkinter import ttk
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui
import joblib
import numpy as np

# Load trained ML model
model = joblib.load("gesture_model.pkl")

LABELS = {
    "0": "OPEN PALM",
    "1": "FIST",
    "2": "PEACE",
    "3": "THUMBS UP",
    "4": "ONE FINGER"
}

# App state 
camera_thread = None
stop_event = threading.Event()
controls_enabled = False
camera_running = False

prediction_history = deque(maxlen=5)
last_action_gesture = None
last_action_time = 0
last_gesture = "NO HAND"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


# ---------- GUI ----------
root = tk.Tk()
root.title("Hand Gesture Recognition - ML Control")
root.geometry("620x500")
root.configure(bg="#101827")

title = tk.Label(
    root,
    text="HAND GESTURE RECOGNITION",
    font=("Segoe UI", 17, "bold"),
    bg="#101827",
    fg="white"
)
title.pack(pady=18)

model_status = tk.Label(
    root,
    text="ML Model: gesture_model.pkl loaded",
    font=("Segoe UI", 10),
    bg="#101827",
    fg="#6ee7b7"
)
model_status.pack()

gesture_label = tk.Label(
    root,
    text="Gesture: Waiting",
    font=("Segoe UI", 15, "bold"),
    bg="#101827",
    fg="white"
)
gesture_label.pack(pady=(20, 5))

action_label = tk.Label(
    root,
    text="Action: None",
    font=("Segoe UI", 12),
    bg="#101827",
    fg="#93c5fd"
)
action_label.pack(pady=5)

status_label = tk.Label(
    root,
    text="Camera stopped | Controls disabled",
    font=("Segoe UI", 10),
    bg="#101827",
    fg="#fbbf24"
)
status_label.pack(pady=10)


def update_gui(gesture=None, action=None, status=None):
    if gesture is not None:
        gesture_label.config(text=f"Gesture: {gesture}")
    if action is not None:
        action_label.config(text=f"Action: {action}")
    if status is not None:
        status_label.config(text=status)


#  Gesture actions 
def perform_action(gesture):
    global last_action_gesture, last_action_time

    if not controls_enabled:
        return

    now = time.time()

    # Don't repeat a click/key while the same gesture is held.
    if gesture == last_action_gesture:
        return

    # Small cooldown between actions.
    if now - last_action_time < 0.6:
        return

    try:
        if gesture == "ONE FINGER":
            # Pointer movement is handled continuously below.
            update_gui(action="Move mouse")

        elif gesture == "FIST":
            pyautogui.click()
            update_gui(action="Left click")

        elif gesture == "PEACE":
            pyautogui.press("right")
            update_gui(action="Next slide / Right Arrow")

        elif gesture == "THUMBS UP":
            pyautogui.press("left")
            update_gui(action="Previous slide / Left Arrow")

        elif gesture == "OPEN PALM":
            pyautogui.press("playpause")
            update_gui(action="Media Play/Pause")

        else:
            return

        last_action_gesture = gesture
        last_action_time = now

    except pyautogui.FailSafeException:
        emergency_stop()
    except Exception as error:
        update_gui(action=f"Action error: {error}")


def toggle_controls():
    global controls_enabled, last_action_gesture

    controls_enabled = not controls_enabled
    last_action_gesture = None

    if controls_enabled:
        controls_button.config(text="Disable Controls")
        update_gui(
            status="Controls ENABLED - show gestures carefully"
        )
    else:
        controls_button.config(text="Enable Controls")
        update_gui(
            status="Controls disabled - camera may still run"
        )


def emergency_stop():
    global controls_enabled, last_action_gesture

    controls_enabled = False
    last_action_gesture = None
    controls_button.config(text="Enable Controls")
    update_gui(
        action="Emergency stop activated",
        status="Controls DISABLED"
    )


#  Camera and ML prediction 
def camera_loop():
    global camera_running, last_gesture, last_action_gesture

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        root.after(
            0,
            lambda: update_gui(status="ERROR: Cannot open webcam")
        )
        camera_running = False
        root.after(0, refresh_buttons)
        return

    prediction_history.clear()
    last_gesture = "NO HAND"
    last_action_gesture = None

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:

            while not stop_event.is_set():
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                current_gesture = "NO HAND"
                confidence = 0.0

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    features = []
                    for point in hand.landmark:
                        features.extend([
                            point.x,
                            point.y,
                            point.z
                        ])

                    # Same 63 features used during training.
                    probabilities = model.predict_proba([features])[0]
                    best_index = int(np.argmax(probabilities))
                    predicted_id = str(model.classes_[best_index])

                    current_gesture = LABELS.get(
                        predicted_id, predicted_id
                    )
                    confidence = float(probabilities[best_index]) * 100

                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    # Smooth predictions across the last 5 frames.
                    prediction_history.append(current_gesture)
                    stable_gesture = current_gesture

                    if (
                        len(prediction_history) == prediction_history.maxlen
                        and len(set(prediction_history)) == 1
                    ):
                        stable_gesture = prediction_history[0]
                    else:
                        stable_gesture = "DETECTING..."

                    if stable_gesture != "DETECTING...":
                        if stable_gesture != last_gesture:
                            last_action_gesture = None
                            last_gesture = stable_gesture

                        # Move the pointer while pointing with one finger.
                        if (
                            controls_enabled
                            and stable_gesture == "ONE FINGER"
                        ):
                            try:
                                x = int(hand.landmark[8].x * pyautogui.size().width)
                                y = int(hand.landmark[8].y * pyautogui.size().height)
                                pyautogui.moveTo(x, y, duration=0.05)
                            except pyautogui.FailSafeException:
                                root.after(0, emergency_stop)

                        perform_action(stable_gesture)

                    cv2.putText(
                        frame,
                        f"Gesture: {stable_gesture}",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"Model score: {confidence:.1f}%",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2
                    )

                    root.after(
                        0,
                        lambda g=stable_gesture, c=confidence:
                        update_gui(
                            gesture=g,
                            status=(
                                f"Camera running | "
                                f"Controls {'ENABLED' if controls_enabled else 'disabled'}"
                            )
                        )
                    )

                else:
                    prediction_history.clear()
                    last_gesture = "NO HAND"
                    last_action_gesture = None
                    root.after(
                        0,
                        lambda: update_gui(gesture="NO HAND")
                    )

                cv2.imshow("ML Gesture Camera - Press Q to close", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    except Exception as error:
        root.after(
            0,
            lambda e=str(error): update_gui(
                status=f"Camera error: {e}"
            )
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        camera_running = False
        root.after(0, refresh_buttons)
        root.after(
            0,
            lambda: update_gui(status="Camera stopped")
        )


def start_camera():
    global camera_thread, camera_running

    if camera_running:
        return

    stop_event.clear()
    camera_running = True
    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )
    camera_thread.start()
    refresh_buttons()
    update_gui(status="Starting camera...")


def stop_camera():
    stop_event.set()
    update_gui(status="Stopping camera...")


def refresh_buttons():
    start_button.config(state="disabled" if camera_running else "normal")
    stop_button.config(state="normal" if camera_running else "disabled")


def close_app():
    stop_event.set()
    root.destroy()


# ---------- Dashboard buttons ----------
button_frame = tk.Frame(root, bg="#101827")
button_frame.pack(pady=22)

start_button = ttk.Button(
    button_frame, text="Start Camera", command=start_camera
)
start_button.grid(row=0, column=0, padx=6, pady=6)

stop_button = ttk.Button(
    button_frame, text="Stop Camera", command=stop_camera
)
stop_button.grid(row=0, column=1, padx=6, pady=6)

controls_button = ttk.Button(
    button_frame, text="Enable Controls", command=toggle_controls
)
controls_button.grid(row=1, column=0, padx=6, pady=6)

emergency_button = ttk.Button(
    button_frame, text="Emergency Stop", command=emergency_stop
)
emergency_button.grid(row=1, column=1, padx=6, pady=6)

stop_button.config(state="disabled")

instructions = tk.Label(
    root,
    text=(
        "ONE FINGER: Move mouse | FIST: Left click\n"
        "PEACE: Next slide | THUMBS UP: Previous slide\n"
        "OPEN PALM: Media Play/Pause\n"
        "Controls start disabled. Test with controls OFF first."
    ),
    font=("Segoe UI", 10),
    bg="#101827",
    fg="#cbd5e1",
    justify="center"
)
instructions.pack(pady=8)

root.protocol("WM_DELETE_WINDOW", close_app)
root.mainloop()
