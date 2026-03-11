import json
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple


HAND_LABELS = ("Left", "Right")


def _normalize_hand(points: List[List[float]]) -> List[float]:
    if not points:
        return [0.0] * (21 * 3)

    wrist = points[0]
    translated = []
    max_distance = 0.0

    for x, y, z in points:
        px = x - wrist[0]
        py = y - wrist[1]
        pz = z - wrist[2]
        translated.append((px, py, pz))
        max_distance = max(max_distance, (px * px + py * py + pz * pz) ** 0.5)

    scale = max_distance or 1.0
    flattened: List[float] = []

    for px, py, pz in translated:
        flattened.extend((px / scale, py / scale, pz / scale))

    return flattened


def _frame_to_vector(frame: Dict[str, List[List[float]]]) -> List[float]:
    vector: List[float] = []

    for label in HAND_LABELS:
        vector.extend(_normalize_hand(frame.get(label, [])))

    return vector


def _sample_frames(frames: List[Dict[str, List[List[float]]]], target_count: int) -> List[List[float]]:
    if not frames:
        return [_frame_to_vector({}) for _ in range(target_count)]

    sampled_vectors: List[List[float]] = []

    for index in range(target_count):
        source_index = int(index * len(frames) / target_count)
        source_index = min(source_index, len(frames) - 1)
        sampled_vectors.append(_frame_to_vector(frames[source_index]))

    return sampled_vectors


def _sequence_distance(left: List[List[float]], right: List[List[float]]) -> float:
    frame_count = min(len(left), len(right))
    if frame_count == 0:
        return float("inf")

    total = 0.0

    for left_frame, right_frame in zip(left[:frame_count], right[:frame_count]):
        total += sum((a - b) ** 2 for a, b in zip(left_frame, right_frame)) ** 0.5

    return total / frame_count


class GestureRecognizer:
    def __init__(
        self,
        data_dir: str = "data",
        window_size: int = 12,
        sample_size: int = 8,
        match_threshold: float = 2.6,
        stable_count: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.sample_size = sample_size
        self.match_threshold = match_threshold
        self.stable_count = stable_count

        self.live_frames: Deque[Dict[str, List[List[float]]]] = deque(maxlen=window_size)
        self.prediction_history: Deque[str] = deque(maxlen=stable_count)
        self.templates: List[Dict[str, object]] = []

        self.reload()

    def reload(self):
        self.templates = []

        if not self.data_dir.exists():
            return

        for sample_path in sorted(self.data_dir.glob("*/*.json")):
            with sample_path.open() as handle:
                payload = json.load(handle)

            label = payload.get("label")
            frames = payload.get("frames", [])
            if not label or not frames:
                continue

            self.templates.append(
                {
                    "label": label,
                    "path": str(sample_path),
                    "sequence": _sample_frames(frames, self.sample_size),
                }
            )

    def update(self, hands) -> Tuple[Optional[str], Optional[float], bool]:
        frame: Dict[str, List[List[float]]] = {}

        for hand in hands:
            frame[hand["label"]] = [[lm.x, lm.y, lm.z] for lm in hand["landmarks"]]

        self.live_frames.append(frame)

        if len(self.live_frames) < self.sample_size or not self.templates:
            return None, None, False

        live_sequence = _sample_frames(list(self.live_frames), self.sample_size)

        best_label = None
        best_score = float("inf")

        for template in self.templates:
            score = _sequence_distance(live_sequence, template["sequence"])
            if score < best_score:
                best_score = score
                best_label = template["label"]

        if best_label is None or best_score > self.match_threshold:
            self.prediction_history.clear()
            return None, best_score, False

        self.prediction_history.append(best_label)
        stable_label, count = Counter(self.prediction_history).most_common(1)[0]
        is_stable = count == self.stable_count and len(self.prediction_history) == self.stable_count

        return stable_label, best_score, is_stable
