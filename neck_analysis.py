import cv2
import numpy as np
import time
from datetime import datetime
import winsound
import os

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmark,
    PoseLandmarksConnections,
    drawing_utils,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


class PostureAnalyzer:
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
        self.PoseLandmark = PoseLandmark
        self.POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS

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

    def calculate_angle(self, a, b, c):
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

    def detect_forward_head(self, landmarks):
        ear = [
            landmarks[self.PoseLandmark.LEFT_EAR.value].x,
            landmarks[self.PoseLandmark.LEFT_EAR.value].y,
        ]
        shoulder = [
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        hip = [
            landmarks[self.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.PoseLandmark.LEFT_HIP.value].y,
        ]

        angle = self.calculate_angle(ear, shoulder, hip)
        return angle < (90 - self.thresholds["forward_head"]), angle

    def detect_slouched_shoulders(self, landmarks):
        left_shoulder = [
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        right_shoulder = [
            landmarks[self.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[self.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        left_hip = [
            landmarks[self.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.PoseLandmark.LEFT_HIP.value].y,
        ]
        right_hip = [
            landmarks[self.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[self.PoseLandmark.RIGHT_HIP.value].y,
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
        return angle > self.thresholds["slouched_shoulders"], angle

    def detect_text_neck(self, landmarks):
        nose = [
            landmarks[self.PoseLandmark.NOSE.value].x,
            landmarks[self.PoseLandmark.NOSE.value].y,
        ]
        shoulder = [
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[self.PoseLandmark.LEFT_SHOULDER.value].y,
        ]

        distance = abs(nose[0] - shoulder[0])
        return distance > self.thresholds["text_neck"] / 100, distance

    def draw_posture_info(self, frame, posture_status, angles):
        h, w = frame.shape[:2]

        green = (0, 255, 0)
        red = (0, 0, 255)
        orange = (0, 165, 255)
        white = (255, 255, 255)

        y_offset = 30
        for i, (key, is_bad) in enumerate(posture_status.items()):
            color = red if is_bad else green
            text = f"{key.replace('_', ' ').title()}: {'MALA' if is_bad else 'Buena'}"
            if key == "forward_head" and angles.get(key):
                text += f" ({angles[key]:.1f}˚)"
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        if self.alert_active:
            alert_text = "!POSTURA INCORRECTA! Corrigete"
            cv2.putText(
                frame,
                alert_text,
                (w // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                red,
                3,
            )
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), red, 3)

        cv2.putText(
            frame,
            f"Tiempo mala postura: {self.posture_stats['bad_posture_time']:.1f}s",
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            white,
            1,
        )
        cv2.putText(
            frame,
            f"Total tiempo: {self.posture_stats['total_time']:.1f}s",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            white,
            1,
        )

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

            drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=self.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                ),
            )

            is_bad, angle = self.detect_forward_head(landmarks)
            posture_status["forward_head"] = is_bad
            angles["forward_head"] = angle

            is_bad, angle = self.detect_slouched_shoulders(landmarks)
            posture_status["slouched_shoulders"] = is_bad
            angles["slouched_shoulders"] = angle

            is_bad, distance = self.detect_text_neck(landmarks)
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

        self.draw_posture_info(frame, posture_status, angles)

        if self.alert_active and self.bad_posture_start_time:
            self.posture_stats["bad_posture_time"] = (
                time.time() - self.bad_posture_start_time
            )

        return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    analyzer = PostureAnalyzer()

    cv2.namedWindow("Posture Analyzer", cv2.WINDOW_NORMAL)

    print("=== Analizador de Postura ===")
    print("Mantén buena postura durante el trabajo")
    print("Presiona 'q' para salir")
    print("Presiona 'r' para resetear estadisticas")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = analyzer.process_frame(frame)
        analyzer.posture_stats["total_time"] = time.time() - start_time

        cv2.imshow("Posture Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            analyzer.posture_stats = {
                "total_time": analyzer.posture_stats["total_time"],
                "bad_posture_time": 0,
                "bad_posture_count": 0,
            }
            start_time = time.time()
            print("Estadisticas reseteadas")

    print("\n=== Reporte Final ===")
    print(f"Tiempo total: {analyzer.posture_stats['total_time']:.1f} segundos")
    print(
        f"Tiempo con mala postura: {analyzer.posture_stats['bad_posture_time']:.1f} segundos"
    )
    print(f"Alertas de mala postura: {analyzer.posture_stats['bad_posture_count']}")

    if analyzer.posture_stats["total_time"] > 0:
        percentage_bad = (
            analyzer.posture_stats["bad_posture_time"]
            / analyzer.posture_stats["total_time"]
            * 100
        )
        print(f"Porcentaje de mala postura: {percentage_bad:.1f}%")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
