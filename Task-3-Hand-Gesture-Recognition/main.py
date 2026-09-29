"""Clean, compact six-gesture hand recognition UI."""
import argparse
import time
from collections import Counter, deque
import cv2
from gesture_detector import HandGestureDetector

GESTURES = ["THUMBS UP", "FIST", "PEACE", "THUMBS DOWN", "POINTING", "OPEN PALM"]

class GestureSmoother:
    def __init__(self, window_size=5, min_votes=3):
        self.history = deque(maxlen=window_size)
        self.min_votes = min_votes

    def update(self, label):
        self.history.append(label)
        if not self.history:
            return "SHOW YOUR HAND"
        name, votes = Counter(self.history).most_common(1)[0]
        return name if votes >= self.min_votes else "DETECTING..."

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal six-gesture hand recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--confidence", type=float, default=0.65)
    return parser.parse_args()

def rounded_panel(frame, x1, y1, x2, y2, color=(22, 28, 43), alpha=0.78):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def draw_ui(frame, active, fps):
    h, w = frame.shape[:2]
    # Compact top banner
    rounded_panel(frame, 14, 14, min(w-14, 430), 94, alpha=0.76)
    cv2.putText(frame, "HAND GESTURE RECOGNITION", (30, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 255), 2, cv2.LINE_AA)
    label = active if active else "SHOW YOUR HAND"
    cv2.putText(frame, label, (30, 76), cv2.FONT_HERSHEY_DUPLEX,
                0.82, (95, 245, 155), 2, cv2.LINE_AA)

    # Tiny FPS badge
    rounded_panel(frame, max(14, w-120), 14, w-14, 54, alpha=0.72)
    cv2.putText(frame, f"{fps:.0f} FPS", (max(25, w-106), 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 240, 250), 1, cv2.LINE_AA)

    # Bottom strip: six compact chips, no large side panel over the hand.
    margin = 14
    gap = 7
    available = w - 2*margin - gap*(len(GESTURES)-1)
    chip_w = max(80, available // len(GESTURES))
    chip_h = 42
    y1 = h - chip_h - 14
    for idx, name in enumerate(GESTURES):
        x1 = margin + idx*(chip_w+gap)
        x2 = min(w-margin, x1+chip_w)
        is_active = name == active
        fill = (35, 145, 90) if is_active else (22, 28, 43)
        alpha = 0.90 if is_active else 0.72
        rounded_panel(frame, x1, y1, x2, y1+chip_h, color=fill, alpha=alpha)
        text = name
        scale = 0.42 if len(text) > 9 else 0.48
        thickness = 2 if is_active else 1
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        tx = x1 + max(4, (x2-x1-tw)//2)
        cv2.putText(frame, text, (tx, y1+27), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255,255,255), thickness, cv2.LINE_AA)

def main():
    args = parse_args()
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Check camera permissions or try --camera 1.")
    # Request a moderate full-frame resolution rather than cropping/zooming.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    detector = HandGestureDetector(max_hands=1, detection_confidence=args.confidence)
    smoother = GestureSmoother()
    previous = time.perf_counter()
    fps = 0.0
    stable = "SHOW YOUR HAND"
    window_name = "Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Unable to read webcam frame.")
                break
            # Mirror selfie view, preserving the entire camera frame.
            frame = cv2.flip(frame, 1)
            frame, hands = detector.process(frame)
            raw = hands[0].label if hands else "SHOW YOUR HAND"
            stable = smoother.update(raw)
            now = time.perf_counter()
            dt = now - previous
            previous = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            draw_ui(frame, stable, fps)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
