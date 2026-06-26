import os
import cv2
import dlib
import numpy as np

PREDICTOR_PATH = os.environ.get(
    "DLIB_PREDICTOR_PATH",
    os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
)

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) != 1:
        return None
    return np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, (x, y) in enumerate(landmarks):
        cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
    return img


def eye_aspect_ratio(landmarks, eye_indices):
    pts = landmarks[eye_indices]
    top = pts[[1, 2]].mean(axis=0)
    bottom = pts[[5, 4]].mean(axis=0)
    left, right = pts[0], pts[3]
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(right - left)
    return vertical / (horizontal + 1e-6)


LEFT_EYE = [36, 37, 38, 39, 40, 41]
WINK_EAR_THRESHOLD = 0.2

cap = cv2.VideoCapture(0)
winks = 0
wink_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_landmarks(frame)
    if landmarks is not None:
        ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        frame = annotate_landmarks(frame, landmarks)

        prev = wink_active
        wink_active = ear < WINK_EAR_THRESHOLD

        if prev and not wink_active:
            winks += 1

        label = f"Wink Count: {winks}"
        color = (0, 0, 255) if wink_active else (0, 255, 127)
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        if wink_active:
            cv2.putText(frame, "Winking!", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Wink Detection", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()