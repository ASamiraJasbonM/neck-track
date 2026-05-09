# Plan de Refactorización: Modularización de neck_analysis.py

## Objetivo
Dividir el monolito `neck_analysis.py` en 5 módulos dentro de `src/`, separando responsabilidades: captura de video, cálculos geométricos, detección MediaPipe, overlay visual, y orquestación.

## Estructura final

```
cuello-track/
├── src/
│   ├── cam.py              # (1) Clase Camera: gestiona cv2.VideoCapture
│   ├── calculos_pos.py     # (2) Funciones puras de cálculo de ángulos
│   ├── pose_analysis.py    # (3) Clase PoseAnalyzer: lógica MediaPipe
│   ├── decorador.py        # (4) Decorador/overlay para dibujar en frame
│   └── main.py             # (5) Orquestador: une todo
├── neck_analysis.py        # ← se conserva o elimina después
├── main.py                 # ← punto de entrada raíz (actualizar import)
└── pose_landmarker_lite.task
```

## Paso a paso

### Paso 1: `src/cam.py` — Clase Camera
- Extraer la lógica de `cv2.VideoCapture` del `main()` actual.
- Crear clase `Camera` con métodos:
  - `__init__(self, camera_id=0, width=640, height=480)`
  - `start()` — abre la cámara, configura propiedades
  - `read()` — captura un frame
  - `release()` — libera recursos
- Constantes: ancho/alto por defecto.

### Paso 2: `src/calculos_pos.py` — Funciones de cálculo
- Extraer funciones puras (sin dependencia de MediaPipe ni cv2):
  - `calculate_angle(a, b, c)` — ángulo entre 3 puntos (ya existe).
  - `detect_forward_head(landmarks, threshold)` — devuelve `(is_bad, angle)`.
  - `detect_slouched_shoulders(landmarks, threshold)` — devuelve `(is_bad, angle)`.
  - `detect_text_neck(landmarks, threshold)` — devuelve `(is_bad, distance)`.
- Reciben landmarks como lista/indexable y un `threshold`.
- No importan MediaPipe ni cv2. Solo `numpy`.

### Paso 3: `src/pose_analysis.py` — Clase PoseAnalyzer
- Extraer la lógica MediaPipe de `PostureAnalyzer` actual.
- Clase `PoseAnalyzer`:
  - `__init__(self, model_path)` — crea el `PoseLandmarker`.
  - `process_frame(self, frame)` — recibe un frame BGR, ejecuta detección, llama a funciones de `calculos_pos`, retorna `(posture_status, angles, landmarks)`.
  - Propiedades de umbrales (`thresholds`).
  - Lógica de alertas y estadísticas (tiempo mala postura, conteo).
  - NO incluye drawing (eso va en decorador).
  - NO incluye captura de cámara (eso va en cam.py).
- Importa `calculos_pos` para los cálculos.

### Paso 4: `src/decorador.py` — Decorador de overlay visual
- Implementar un decorador Python (o función de orden superior) que envuelva una función de procesamiento de frames para añadir el overlay visual.
- Alternativa: clase `PostureOverlay` con método `apply(frame, posture_status, angles, alert_active, stats)`.
- Extraer la lógica actual de `draw_posture_info` y las alertas visuales.
- Podría ser algo como:
  ```python
  @posture_overlay
  def process_and_display(frame, analyzer):
      ...
  ```
- Maneja: textos de estado, rectángulo de alerta, estadísticas en pantalla.

### Paso 5: `src/main.py` — Orquestador
- Importa `Camera` de `cam.py`.
- Importa `PoseAnalyzer` de `pose_analysis.py`.
- Importa decorador/overlay de `decorador.py`.
- Bucle principal:
  1. Inicializar Camera.
  2. Inicializar PoseAnalyzer.
  3. Bucle: leer frame → analizar → aplicar overlay → mostrar.
  4. Tecla 'q' para salir, 'r' para resetear stats.
  5. Reporte final al salir.

### Paso 6: Actualizar `main.py` raíz
- Cambiar import de `neck_analysis` a `src.main`.

### Paso 7 (opcional): Tests
- Verificar que `src/calculos_pos.py` se pueda testear unitariamente (funciones puras).
- Verificar que los imports circulares no existan.

## Dependencias entre módulos
```
main.py → src.main
src.main → src.cam, src.pose_analysis, src.decorador
src.pose_analysis → src.calculos_pos
src.decorador → (solo cv2)
src.calculos_pos → numpy
src.cam → cv2
```

## Notas
- `pose_landmarker_lite.task` se queda en la raíz; `PoseAnalyzer` recibe la ruta como parámetro.
- Los thresholds se definen en `pose_analysis.py` (o podrían ir a un config).
- El decorador puede ser un patrón de función que retorna una función-wrapper, o una clase con método `__call__`.
