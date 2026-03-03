import os
import json
import time

class AlphabetRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.label = None

    def start(self, label):
        self.recording = True
        self.frames = []
        self.label = label
        print(f"[REC] Recording started for: {label}")

    def stop(self):
        if not self.recording:
            return

        self.recording = False
        print(f"[REC] Recording stopped. Saving...")

        os.makedirs(f"data/{self.label}", exist_ok=True)

        timestamp = int(time.time())
        filename = f"data/{self.label}/{timestamp}.json"

        with open(filename, "w") as f:
            json.dump({
                "label": self.label,
                "frames": self.frames
            }, f)

        print(f"[REC] Saved to {filename}")

    def add_frame(self, hands):
        if not self.recording:
            return

        frame_data = {}

        for hand in hands:
            coords = []
            for lm in hand["landmarks"]:
                coords.append([lm.x, lm.y, lm.z])

            frame_data[hand["label"]] = coords

        self.frames.append(frame_data)