# Facial Emotion Detection with MediaPipe and OpenCV

A computer vision application that detects human emotions using MediaPipe and OpenCV. This project includes two implementations: real-time detection from webcam feed and static image analysis.

## Features

- **Emotion Detection**: Recognizes 7 basic emotions
  - Happiness
  - Sadness
  - Anger
  - Surprise
  - Fear
  - Disgust
  - Neutral


- **Two Modes**:
  - **Live Detection** (`mediapipe_cv_live.py`): Real-time analysis from webcam
  - **Static Detection** (`mediapipe_cv_static.py`): Analysis of static images

## Requirements

- Python 3.7 or higher
- Webcam (for live detection mode)

## Installation

1. Clone this repository:
```
git clone https://github.com/samikshapatil08/CV_mediapipe_blog
cd CV_mediapipe_blog
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Live Detection (Webcam)

Run the live detection script to analyze video from your webcam:

```bash
python mediapipe_cv_live.py
```

**Controls:**
- Press `Q` or `ESC` to quit the application
- The FPS counter displays real-time performance
- Emotion label updates dynamically based on facial expressions

### Static Image Detection

Run the static detection script to analyze a single image:

```bash
python mediapipe_cv_static.py
```

**Note:** Update the `image_path` variable in the script to point to your image:
```python
image_path = "your_image.jpg"  # Change this to your image path
```

Press any key to close the window after viewing the results.

## How It Works
read here: https://medium.com/@samikshapatil486/can-your-camera-tell-if-youre-bored-in-class-bfece6871e58


## Customization

### Adjusting Emotion Thresholds

You can fine-tune emotion detection by modifying thresholds in the `detect_emotion()` function:

```python
if smile_score > 0.05:  # Adjust this value
    return "Happiness"
```

### Changing Detection Confidence

Modify confidence thresholds in the initialization:

```python
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # Adjust detection sensitivity
    min_tracking_confidence=0.5     # Adjust tracking stability
)
```

## Limitations

- Emotion detection is based on geometric features and may not be as accurate as deep learning models
- Works best with frontal face views and good lighting
- Single face detection only
- Emotion classification uses heuristic rules rather than trained models

## Troubleshooting

**Camera not opening:**
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

**Low FPS:**
- Close other resource-intensive applications
- Reduce camera resolution if needed
- Consider using a more powerful GPU

**Image not loading (static mode):**
- Verify the image path is correct
- Ensure the image format is supported (JPG, PNG, etc.)


## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for pose and face mesh solutions
- [OpenCV](https://opencv.org/) for computer vision utilities

