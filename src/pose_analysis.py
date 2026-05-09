import os
import time

import cv2
import winsound
from mediapipe import ImageFormat
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python._framework_bindings.image import Image
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from src.calculos_pos import (
    detect_forward_head,
    detect_slouched_shoulders,
    detect_text_neck,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task")


class PoseAnalyzer:
    def __init__(self, model_path: str = MODEL_PATH):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            num_poses=1,
        )
        self.detector = PoseLandmarker.create_from_options(options)
        self.POSE_CONNECTIONS = [
            (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
        ]

        self.bad_posture_start_time = None
        self.alert_cooldown = 0
        self.alert_active = False

        self.thresholds = {
            "forward_head": 15.0,
            "slouched_shoulders": 25.0,
            "text_neck": 20.0,
        }

        self.posture_stats = {
            "total_time": 0,
            "bad_posture_time": 0,
            "bad_posture_count": 0,
        }

    def trigger_alert(self):
        self.alert_active = True
        self.bad_posture_start_time = self.bad_posture_start_time or time.time()

        try:
            winsound.Beep(1000, 500)
        except Exception:
            pass

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(image)

        posture_status = {
            "forward_head": False,
            "slouched_shoulders": False,
            "text_neck": False,
        }
        angles = {}

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in landmarks:
                entry = landmark_proto.landmark.add()
                entry.x = lm.x
                entry.y = lm.y
                entry.z = lm.z
                if lm.visibility is not None:
                    entry.visibility = lm.visibility
                if lm.presence is not None:
                    entry.presence = lm.presence

            draw_landmarks(
                image=frame,
                landmark_list=landmark_proto,
                connections=self.POSE_CONNECTIONS,
                landmark_drawing_spec=DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            is_bad, angle = detect_forward_head(
                landmarks, self.thresholds["forward_head"]
            )
            posture_status["forward_head"] = is_bad
            angles["forward_head"] = angle

            is_bad, angle = detect_slouched_shoulders(
                landmarks, self.thresholds["slouched_shoulders"]
            )
            posture_status["slouched_shoulders"] = is_bad
            angles["slouched_shoulders"] = angle

            is_bad, distance = detect_text_neck(landmarks, self.thresholds["text_neck"])
            posture_status["text_neck"] = is_bad
            angles["text_neck"] = distance

        any_bad = any(posture_status.values())

        if any_bad:
            if self.bad_posture_start_time is None:
                self.bad_posture_start_time = time.time()

            if time.time() - self.bad_posture_start_time > 5:
                self.trigger_alert()
                self.posture_stats["bad_posture_count"] += 1
        else:
            self.bad_posture_start_time = None
            self.alert_active = False

        if self.alert_active and self.bad_posture_start_time:
            self.posture_stats["bad_posture_time"] = (
                time.time() - self.bad_posture_start_time
            )

        return frame, posture_status, angles
