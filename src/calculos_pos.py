import numpy as np

LEFT_EAR = 7
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def detect_forward_head(landmarks, threshold):
    ear = [
        landmarks[LEFT_EAR].x,
        landmarks[LEFT_EAR].y,
    ]
    shoulder = [
        landmarks[LEFT_SHOULDER].x,
        landmarks[LEFT_SHOULDER].y,
    ]
    hip = [
        landmarks[LEFT_HIP].x,
        landmarks[LEFT_HIP].y,
    ]

    angle = calculate_angle(ear, shoulder, hip)
    return angle < (90 - threshold), angle


def detect_slouched_shoulders(landmarks, threshold):
    left_shoulder = [
        landmarks[LEFT_SHOULDER].x,
        landmarks[LEFT_SHOULDER].y,
    ]
    right_shoulder = [
        landmarks[RIGHT_SHOULDER].x,
        landmarks[RIGHT_SHOULDER].y,
    ]
    left_hip = [
        landmarks[LEFT_HIP].x,
        landmarks[LEFT_HIP].y,
    ]
    right_hip = [
        landmarks[RIGHT_HIP].x,
        landmarks[RIGHT_HIP].y,
    ]

    shoulder_center = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    ]
    hip_center = [
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    ]

    angle = abs(shoulder_center[0] - hip_center[0]) * 100
    return angle > threshold, angle


def detect_text_neck(landmarks, threshold):
    nose = [
        landmarks[NOSE].x,
        landmarks[NOSE].y,
    ]
    shoulder = [
        landmarks[LEFT_SHOULDER].x,
        landmarks[LEFT_SHOULDER].y,
    ]

    distance = abs(nose[0] - shoulder[0])
    return distance > threshold / 100, distance
