import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path: str = "hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_result = None

    def get_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.last_result = self.detector.detect(mp_image)

        hands_data = []

        if not self.last_result.hand_landmarks:
            return hands_data

        for i, hand_landmarks in enumerate(self.last_result.hand_landmarks):
            label = self.last_result.handedness[i][0].category_name

            # 🔹 라벨 반전
            if label == "Left":
                label = "Right"
            else:
                label = "Left"

            hands_data.append({
                "label": label,
                "landmarks": hand_landmarks
            })

        return hands_data

    def draw(self, frame):   # 🔥 클래스 안에 있어야 함
        if not self.last_result or not self.last_result.hand_landmarks:
            return

        h, w, _ = frame.shape

        for i, hand_landmarks in enumerate(self.last_result.hand_landmarks):
            label = self.last_result.handedness[i][0].category_name

            # 🔹 여기에도 반전 적용
            if label == "Left":
                label = "Right"
            else:
                label = "Left"

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            wrist = hand_landmarks[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, label, (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)