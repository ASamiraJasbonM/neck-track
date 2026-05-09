import functools

import cv2


class PostureOverlay:
    def __init__(self):
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.orange = (0, 165, 255)
        self.white = (255, 255, 255)

    def draw(self, frame, posture_status, angles, alert_active, stats):
        h, w = frame.shape[:2]

        y_offset = 30
        for i, (key, is_bad) in enumerate(posture_status.items()):
            color = self.red if is_bad else self.green
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

        if alert_active:
            alert_text = "!POSTURA INCORRECTA! Corrigete"
            cv2.putText(
                frame,
                alert_text,
                (w // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.red,
                3,
            )
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), self.red, 3)

        cv2.putText(
            frame,
            f"Tiempo mala postura: {stats['bad_posture_time']:.1f}s",
            (10, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.white,
            1,
        )
        cv2.putText(
            frame,
            f"Total tiempo: {stats['total_time']:.1f}s",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.white,
            1,
        )

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(frame, analyzer, *args, **kwargs):
            result = func(frame, analyzer, *args, **kwargs)
            if result is not None:
                frame, posture_status, angles = result
                self.draw(
                    frame,
                    posture_status,
                    angles,
                    analyzer.alert_active,
                    analyzer.posture_stats,
                )
            return frame

        return wrapper
