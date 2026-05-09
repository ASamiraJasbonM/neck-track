# neck-track

Analizador de postura cervical en tiempo real usando MediaPipe Pose Landmarker.

## Requisitos del sistema

- Python 3.9+
- Windows (usa `winsound` para alertas sonoras)
- Cámara web

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/neck-track.git
cd neck-track

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py
```

- `q` — salir
- `r` — resetear estadísticas

## Estructura del proyecto

neck-track/
├── src/
│   ├── cam.py              # Clase Camera: captura de video con OpenCV
│   ├── calculos_pos.py     # Funciones de cálculo de ángulos (puras, sin MediaPipe)
│   ├── pose_analysis.py    # Clase PoseAnalyzer: detección MediaPipe + lógica de postura
│   ├── decorador.py        # PostureOverlay: decorador para overlay visual en frames
│   └── main.py             # Orquestador principal
├── main.py                 # Punto de entrada (importa src.main)
├── neck_analysis.py        # Versión monolítica original
├── pose_landmarker_lite.task
├── PLAN.md                 # Plan de arquitectura
└── README.md
