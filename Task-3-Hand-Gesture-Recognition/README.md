# Clean Six-Gesture Hand Gesture Recognition

A minimal webcam application for Syntecxhub AI Internship Task 3.

## Supported gestures
- Thumbs Up
- Fist
- Peace
- Thumbs Down
- Pointing
- Open Palm

## UI
The webcam feed remains the main focus. A compact top banner shows the current gesture, and six small gesture indicators sit along the bottom. The UI avoids a large side panel covering the hand.

## Install and run (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
If your webcam is not camera 0:
```bash
python main.py --camera 1
```

## Keys
- `q` or `Esc`: close the window

## Notes
The application requests 960x540 by default and displays the full camera frame without cropping. If the hand is still too large or small, adjust the physical distance from the webcam or change `--width` / `--height` to a supported camera mode. Classification uses heuristic hand landmarks; results can vary with lighting and hand orientation.
