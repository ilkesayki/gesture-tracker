# 🖐 Gesture tracker - “Monkey Meme from TikTok”
Real-time hand and face gesture detection using Python and OpenCV.
This project uses your webcam to recognize different gestures — like pointing, face proximity, and idle state — and triggers actions or visual changes based on them.

⚙️ Requirements

Before running the project, make sure you have the following installed:
1. Python
You need Python 3.8 or higher. Check your version:
python3 --version
If you don’t have it, install from python.org/downloads.

2. Create and Activate a Virtual Environment
We recommend using a virtual environment (so dependencies don’t conflict):
python3 -m venv ~/venvs/gesture312
source ~/venvs/gesture312/bin/activate

3. Install Dependencies

pip install opencv-python mediapipe numpy absl-py

💻 How to Run
    1    Clone this repository:
        git clone https://github.com/ilkesayki/gesture-tracker.git
        cd gesture-tracker

    2    Activate your virtual environment:
        source ~/venvs/gesture312/bin/activate
    3    Run the script:
        python gesture_tracker.py
    4    If camera access is denied on macOS:
    ⁃    Go to System Settings → Privacy & Security → Camera
    ⁃    Enable access for “Visual Studio Code” or “Terminal”.

🧩 VS Code Setup (Important)
    If you use Visual Studio Code, make sure VS Code uses the correct Python interpreter from your virtual environment.

    Create this file:
    .vscode/settings.json
    
    Add this content:
    {
      "python.defaultInterpreterPath": "/Users/ilkesayki/venvs/gesture312/bin/python"
    }
    
    💡 Without this, VS Code may try to run with the system Python

Features
✅ Detects three main gestures:
    •    ☝️ Point Gesture – when the index finger is up
    •    👃 Face Proximity Gesture – when the finger is close to the face
    •    😶 Idle Gesture – when no significant hand movement is detected
    •    Middle Finger - when the middle finger is up
✅ Real-time feedback with live video display ✅ Multi-camera support (switch between available cameras) ✅ Secondary window displaying different images or actions depending on the gesture
🪄 Customization
    You can fine-tune gesture recognition by adjusting thresholds in the code:
    
    # Example: change proximity sensitivity
    FACE_HAND_RATIO_THRESHOLD = 0.25

    This value determines how close the finger must be to your face to trigger the "proximity" gesture.

🧩 Planned Features
    •    ✋ Add more gesture types 
    •    🪟 Better GUI window for gesture-based photo switching
    •    📷 Gesture-based screenshot capture
    •    🎵 Integration with music or volume control
🧑‍💻 Author
İlke Saykı RWTH Aachen – 2025 💬 Feel free to open an issue or contribute!


