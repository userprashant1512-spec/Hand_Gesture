
import cv2
import mediapipe as mp
import joblib

model = joblib.load("gesture_model.pkl")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

labels = {
    "0": "OPEN PALM",
    "1": "FIST",
    "2": "PEACE",
    "3": "THUMBS UP",
    "4": "ONE FINGER"
}

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = "NO HAND"
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            features = []
            for point in hand.landmark:
                features.extend([point.x, point.y, point.z])

            prediction_id = str(model.predict([features])[0])
            confidence = max(model.predict_proba([features])[0]) * 100
            prediction = labels.get(prediction_id, prediction_id)

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"Gesture: {prediction}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Model score: {confidence:.1f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("Trained Gesture Model Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
