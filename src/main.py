import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cam import Camera
from src.decorador import PostureOverlay
from src.pose_analysis import PoseAnalyzer

overlay = PostureOverlay()


@overlay
def process_with_overlay(frame, analyzer):
    return analyzer.process_frame(frame)


def main():
    cam = Camera()
    cam.start()

    analyzer = PoseAnalyzer()

    cv2.namedWindow("Posture Analyzer", cv2.WINDOW_NORMAL)

    print("=== Analizador de Postura ===")
    print("Mantén buena postura durante el trabajo")
    print("Presiona 'q' para salir")
    print("Presiona 'r' para resetear estadisticas")

    start_time = time.time()

    while True:
        frame = cam.read()
        if frame is None:
            break

        frame = process_with_overlay(frame, analyzer)
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

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
