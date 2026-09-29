# ✋ Real-Time Hand Gesture Recognition and Control System

A real-time hand gesture recognition project built using **Python, OpenCV, and MediaPipe**. The system detects hand landmarks through a webcam and recognizes six hand gestures with a clean, user-friendly interface.

## 📌 Project Overview

The Real-Time Hand Gesture Recognition and Control System uses computer vision to identify hand gestures from a live webcam feed.

It recognizes six basic hand gestures and displays the detected gesture on the screen with a visual indicator.

## ✨ Features

* Real-time hand detection using a webcam.
* Recognition of six hand gestures.
* Clean and user-friendly interface.
* Visual indication of the currently recognized gesture.
* Live camera feed with hand landmark visualization.
* Lightweight implementation using Python.

## 🤚 Supported Gestures

| Gesture        | Description                     |
| -------------- | ------------------------------- |
| 👍 Thumbs Up   | Thumb raised                    |
| ✊ Fist         | All fingers folded              |
| ✌️ Peace       | Index and middle fingers raised |
| 👎 Thumbs Down | Thumb pointing downward         |
| ☝️ Pointing    | Index finger raised             |
| ✋ Open Palm    | All fingers extended            |

## 🛠️ Technologies Used

* **Python** – Core programming language
* **OpenCV** – Image processing and webcam access
* **MediaPipe** – Hand landmark detection
* **NumPy** – Numerical operations

## 📂 Project Structure

```text
Hand-Gesture-Recognition/
│
├── main.py
├── gesture_detector.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Hand-Gesture-Recognition.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Navigate to the Project Folder

```bash
cd Hand-Gesture-Recognition
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```

Allow webcam access when prompted.

## 🚀 How It Works

1. The webcam captures live video frames.
2. MediaPipe detects hand landmarks.
3. The gesture detection logic analyzes the hand landmarks.
4. The system identifies the gesture based on predefined rules.
5. The detected gesture is displayed on the screen.

## 🎯 Applications

* Human-computer interaction
* Touchless interface systems
* Gesture-based control applications
* Computer vision learning projects
* Accessibility-focused interaction systems

## 🔮 Future Enhancements

* Add more hand gestures.
* Integrate gesture-based system controls.
* Improve recognition robustness across different hand positions.
* Explore machine learning-based gesture classification.

## 👩‍💻 Author

**Sumedha Mogallapalli**

B.Tech – Computer Science and Engineering
JNTUA College of Engineering, Kalikiri

## 📜 License

This project is open-source and available for educational and learning purposes.
