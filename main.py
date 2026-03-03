import cv2
from hand_tracker import HandTracker
from alphabet_recorder import AlphabetRecorder

tracker = HandTracker()
cap = cv2.VideoCapture(0)
recorder = AlphabetRecorder()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 1. 모델에는 원본 frame
    hands = tracker.get_hands(frame)

    # 🔹 2. 화면 표시용은 flip
    display = cv2.flip(frame, 1)

    tracker.draw(display)

    cv2.imshow("Hand", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

    recorder.add_frame(hands)

    display = cv2.flip(frame, 1)
    hands = tracker.get_hands(display)
    recorder.add_frame(hands)
    tracker.draw(display)

    cv2.imshow("ASL Recorder", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        label = input("Enter alphabet label (A-Z): ").upper()
        recorder.start(label)
    elif key == ord('s'):
       recorder.stop()

cap.release()
cv2.destroyAllWindows()

