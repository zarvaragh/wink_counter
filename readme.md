# Wink Counter

Real-time wink detection using a webcam, **dlib** facial landmarks, and **OpenCV**. Counts winks using the **Eye Aspect Ratio (EAR)** algorithm — the same technique used in drowsiness detection systems.

## Demo

The camera feed shows annotated facial landmarks in real time. Each completed wink increments the counter displayed on screen.

## Requirements

```bash
pip install opencv-python dlib numpy
```

> **Note:** Installing `dlib` requires CMake and a C++ compiler.  
> On Windows: `pip install cmake` then `pip install dlib`

## Setup

Download the pretrained dlib facial landmark predictor:

- **Download:** [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract and place `shape_predictor_68_face_landmarks.dat` in the same folder as `winks_counter.py`

Or set an environment variable to a custom path:

```bash
# Linux/macOS
export DLIB_PREDICTOR_PATH=/path/to/shape_predictor_68_face_landmarks.dat

# Windows
set DLIB_PREDICTOR_PATH=C:\path\to\shape_predictor_68_face_landmarks.dat
```

## Usage

```bash
python winks_counter.py
```

Press **Enter** to quit.

## How it works

1. Each frame is passed to dlib's HOG-based face detector
2. 68 facial landmarks are predicted for the detected face
3. The **Eye Aspect Ratio (EAR)** is calculated for the left eye:
   - `EAR = vertical_distance / horizontal_distance`
   - When EAR drops below `0.2`, the eye is considered closed (winking)
4. A wink is counted when EAR transitions from below → above the threshold

## Tech Stack

- **dlib** — face detection and 68-point landmark prediction
- **OpenCV** — webcam capture and frame rendering
- **NumPy** — EAR geometry computation