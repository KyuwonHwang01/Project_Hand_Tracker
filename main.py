import cv2
import time
from hand_tracker import HandTracker
from alphabet_recorder import AlphabetRecorder
from gesture_recognizer import GestureRecognizer

tracker = HandTracker()
recorder = AlphabetRecorder()
recognizer = GestureRecognizer()

cap = cv2.VideoCapture(0)
translated_text = ""
last_committed_label = None
last_commit_time = 0.0
commit_cooldown_seconds = 1.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hands = tracker.get_hands(frame)
    recorder.add_frame(hands)
    predicted_label, score, is_stable = recognizer.update(hands)

    if is_stable:
        now = time.time()
        if predicted_label != last_committed_label or (now - last_commit_time) > commit_cooldown_seconds:
            translated_text += predicted_label
            last_committed_label = predicted_label
            last_commit_time = now

    tracker.draw(frame)

    samples_loaded = len(recognizer.templates)
    score_text = "--" if score is None else f"{score:.2f}"
    predicted_text = predicted_label or "-"

    cv2.rectangle(frame, (10, 10), (630, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"Prediction: {predicted_text}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Score: {score_text}  Samples: {samples_loaded}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Text: {translated_text[-24:]}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "r=start s=stop c=clear backspace=delete q=quit", (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("ASL Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        label = input("Enter alphabet label (A-Z): ").upper()
        recorder.start(label)
    elif key == ord('s'):
        recorder.stop()
        recognizer.reload()
    elif key == ord('c'):
        translated_text = ""
        last_committed_label = None
    elif key in (8, 127):
        translated_text = translated_text[:-1]
        last_committed_label = None

cap.release()
cv2.destroyAllWindows()
