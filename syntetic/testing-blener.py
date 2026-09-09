"""
Detección, Clasificación y Anotación de Partículas — Color RGB
===============================================================
Compatible con el tracker (sinker-shadowgraph).

Input:  Carpeta con frames PNG (partículas rojo=slow, verde=medium, azul=fast)
Output:
  - Un CSV por imagen, mismo nombre, misma carpeta
  - Imágenes anotadas en subcarpeta /annotated/

Columnas CSV (compatibles con tracker):
  particle_id, x, xx, y, yy, area, speed_class, timestamp,
  image_path, image_width, image_height

  x, y   = esquina superior izquierda del bounding box
  xx, yy = esquina inferior derecha del bounding box
  area   = área del contorno en píxeles

Dependencias:
  pip install opencv-python numpy pandas
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN ─────────────────────────────────────────────────────────────

INPUT_FOLDER  = "/mbari/Tempbox/fernanda/particles_v3/"
ANNOT_FOLDER  = "/mbari/Tempbox/fernanda/particles_v3/annotated/"

START_DATETIME    = datetime(2024, 1, 1, 0, 0, 0)
SECONDS_PER_FRAME = 1

BACKGROUND_MIN = 200
MIN_AREA_PX    = 0
MAX_AREA_PX    = 800000
COLOR_MARGIN   = 20

IMAGE_WIDTH    = 4600
IMAGE_HEIGHT   = 4000

CLASS_CONFIG = {
    "slow":   {"color": (60,  60,  220)},   # anotación rojo  BGR
    "medium": {"color": (40,  180,  40)},   # anotación verde BGR
    "fast":   {"color": (220,  60,  60)},   # anotación azul  BGR
}

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def frame_to_timestamp(frame_number):
    dt = START_DATETIME + timedelta(seconds=(frame_number - 1) * SECONDS_PER_FRAME)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def extract_frame_number(path):
    digits = ''.join(filter(str.isdigit, path.stem))
    return int(digits) if digits else 1

def classify_color(mean_bgr):
    b, g, r = mean_bgr
    if b > BACKGROUND_MIN and g > BACKGROUND_MIN and r > BACKGROUND_MIN:
        return None
    max_ch = max(r, g, b)
    second = sorted([r, g, b])[-2]
    if max_ch - second < COLOR_MARGIN:
        return None
    if max_ch == r: return "slow"
    if max_ch == g: return "medium"
    if max_ch == b: return "fast"
    return None

# ─── DETECCIÓN ─────────────────────────────────────────────────────────────────

def detect(image_path, timestamp):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    h, w = img.shape[:2]

    not_white = np.any(img < BACKGROUND_MIN, axis=2).astype(np.uint8) * 255
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    not_white = cv2.morphologyEx(not_white, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(not_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    cnt_map    = {}

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        # Centroide
        cx = round(M["m10"] / M["m00"], 2)
        cy = round(M["m01"] / M["m00"], 2)

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(cnt)
        x_min = bx
        y_min = by
        x_max = bx + bw
        y_max = by + bh

        # Color promedio
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_bgr = cv2.mean(img, mask=mask)[:3]

        cls = classify_color(mean_bgr)
        if cls is None:
            continue

        detections.append({
            "particle_id":  len(detections) + 1,
            # centroide (útil para visualización)
            "cx":           cx,
            "cy":           cy,
            # bounding box (requerido por el tracker)
            "x":            x_min,
            "xx":           x_max,
            "y":            y_min,
            "yy":           y_max,
            # área
            "area":         int(area),
            # clase y tiempo
            "speed_class":  cls,
            "timestamp":    timestamp,
            # metadata de imagen (requerido por el tracker)
            "image_path":   str(image_path),
            "image_width":  w,
            "image_height": h,
            "_idx":         idx,
        })
        cnt_map[idx] = cnt

    # ── Anotación ──
    annotated  = img.copy()
    thick      = max(4, w // 1000)
    font_scale = max(1.2, w / 2800)
    font_thick = max(3, thick - 1)

    for d in detections:
        cnt    = cnt_map[d["_idx"]]
        color  = CLASS_CONFIG[d["speed_class"]]["color"]
        cx, cy = int(d["cx"]), int(d["cy"])

        cv2.drawContours(annotated, [cnt], -1, color, thick)

        label = d["speed_class"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        lx = cx - tw // 2
        ly = cy - int(th * 0.9) - 12
        cv2.rectangle(annotated, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 6),
                      (255, 255, 255), -1)
        cv2.putText(annotated, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick)

    for d in detections:
        d.pop("_idx", None)

    return detections, annotated

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def process_folder(input_folder, annot_folder):
    files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    if not files:
        print(f"No se encontraron archivos PNG en: {input_folder}")
        return

    os.makedirs(annot_folder, exist_ok=True)
    print(f"Procesando {len(files)} imágenes")
    print(f"Inicio:    {START_DATETIME.strftime('%Y-%m-%d %H:%M:%S')}  (frame 1)")
    print(f"Fin:       {frame_to_timestamp(len(files))}  (frame {len(files)})")
    print(f"Anotadas → {annot_folder}\n")

    for i, fpath in enumerate(files):
        path         = Path(fpath)
        frame_number = extract_frame_number(path)
        timestamp    = frame_to_timestamp(frame_number)

        detections, annotated = detect(path, timestamp)
        if detections is None:
            print(f"  [!] Error leyendo {path.name}")
            continue

        pd.DataFrame(detections).to_csv(path.with_suffix(".csv"), index=False)
        cv2.imwrite(os.path.join(annot_folder, path.name), annotated)

        if (i + 1) % 50 == 0 or i == 0 or i == len(files) - 1:
            c = {k: sum(1 for d in detections if d["speed_class"] == k)
                 for k in ["slow", "medium", "fast"]}
            print(f"  [{i+1:>5}/{len(files)}]  {timestamp}  "
                  f"slow={c['slow']} medium={c['medium']} fast={c['fast']}")

    print(f"\nListo. {len(files)} CSVs + imágenes anotadas.")
    print(f"Columnas CSV: particle_id, cx, cy, x, xx, y, yy, area, speed_class, timestamp, image_path, image_width, image_height")


if __name__ == "__main__":
    process_folder(INPUT_FOLDER, ANNOT_FOLDER)