"""Focused, heuristic classifier for six requested hand gestures."""
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import cv2
import mediapipe as mp

@dataclass
class GestureResult:
    label: str
    confidence: float
    handedness: str
    landmarks: object
    bbox: Optional[Tuple[int, int, int, int]]

class HandGestureDetector:
    GESTURES = ("THUMBS UP", "FIST", "PEACE", "THUMBS DOWN", "POINTING", "OPEN PALM")

    def __init__(self, max_hands=1, detection_confidence=0.70, tracking_confidence=0.65):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _finger_extended(lm, tip, pip, mcp):
        # Tip should be farther toward the fingertip direction than its PIP joint.
        # The margin reduces false positives caused by small landmark jitter.
        return lm[tip].y < lm[pip].y - 0.025

    def _classify(self, lm):
        index = self._finger_extended(lm, 8, 6, 5)
        middle = self._finger_extended(lm, 12, 10, 9)
        ring = self._finger_extended(lm, 16, 14, 13)
        pinky = self._finger_extended(lm, 20, 18, 17)
        four = [index, middle, ring, pinky]

        # Palm scale normalizes thumb distance to hand size.
        scale = max(self._dist(lm[0], lm[9]), 1e-4)
        thumb_reach = self._dist(lm[4], lm[5]) / scale
        thumb_extended = thumb_reach > 0.72
        curled_count = sum(not state for state in four)

        # Distinguish the requested gestures in priority order.
        if all(four):
            return "OPEN PALM", 0.96
        if not any(four) and not thumb_extended:
            return "FIST", 0.94
        if index and middle and not ring and not pinky:
            return "PEACE", 0.91
        if index and not middle and not ring and not pinky:
            return "POINTING", 0.89

        # Thumbs up/down: four fingers curled and thumb clearly extended vertically.
        if curled_count == 4 and thumb_extended:
            tip_y = lm[4].y
            thumb_mcp_y = lm[2].y
            if tip_y < thumb_mcp_y - 0.055:
                return "THUMBS UP", 0.90
            if tip_y > thumb_mcp_y + 0.055:
                return "THUMBS DOWN", 0.88
        return "UNKNOWN", 0.50

    @staticmethod
    def _bbox(hand_landmarks, width, height):
        xs = [int(p.x * width) for p in hand_landmarks.landmark]
        ys = [int(p.y * height) for p in hand_landmarks.landmark]
        pad = 16
        return (max(0, min(xs)-pad), max(0, min(ys)-pad),
                min(width-1, max(xs)+pad), min(height-1, max(ys)+pad))

    def process(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        output = self.hands.process(rgb)
        results = []
        if output.multi_hand_landmarks:
            handedness_list = output.multi_handedness or []
            for idx, hand_lm in enumerate(output.multi_hand_landmarks):
                handedness = "Unknown"
                if idx < len(handedness_list) and handedness_list[idx].classification:
                    handedness = handedness_list[idx].classification[0].label
                label, confidence = self._classify(hand_lm.landmark)
                bbox = self._bbox(hand_lm, width, height)
                results.append(GestureResult(label, confidence, handedness, hand_lm, bbox))
                self.mp_draw.draw_landmarks(
                    frame_bgr, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style())
                x1, y1, x2, y2 = bbox
                color = (40, 220, 100) if label != "UNKNOWN" else (0, 190, 255)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f"{label} | {handedness} | {confidence:.0%}",
                            (x1, max(25, y1-9)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.60, color, 2, cv2.LINE_AA)
        return frame_bgr, results

    def close(self):
        self.hands.close()
