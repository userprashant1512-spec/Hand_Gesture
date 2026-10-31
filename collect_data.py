
import cv2
import mediapipe as mp
import csv
import os

DATA_FILE = "gesture_data.csv"

gestures = {
    "0": "OPEN_PALM",
    "1": "FIST",
    "2": "PEACE",
    "3": "THUMBS_UP",
    "4": "ONE_FINGER"
}

mp_hands = mp.solutions.hands

file_exists = os.path.exists(DATA_FILE)

with open(DATA_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(
            ["label"] +
            [f"feature_{i}" for i in range(63)]
        )

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        print("Press 0-4 to select a gesture.")
        print("Hold your hand steady to collect samples.")
        print("Press Q to quit.")

        selected = None
        count = 0

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                features = []
                for landmark in hand.landmark:
                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                if selected is not None:
                    writer.writerow([selected] + features)
                    count += 1

                cv2.putText(
                    frame,
                    f"Gesture: {gestures.get(selected, 'NOT SELECTED')}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Samples: {count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if chr(key) in gestures if key < 128 else False:
                selected = chr(key)
                print("Collecting:", gestures[selected])

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

print(f"Saved {count} samples in {DATA_FILE}")
