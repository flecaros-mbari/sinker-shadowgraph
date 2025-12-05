import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
import sys
import pims
from loguru import logger
import numpy as np
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize



# ============================================================
# CLASS: SimpleKalman
# ============================================================
class SimpleKalman:
    def __init__(self, x_min, x_max, y_min, y_max, dt=1.0):
        self.dt = dt
        cx = int(np.mean([x_min, x_max]))
        cy = int(np.mean([y_min, y_max]))
        # estado: [x, y, vx, vy]
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].ravel()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].ravel()

# ============================================================
# CLASS: ParticleTrack
# ============================================================
class ParticleTrack:
    def __init__(self, detection, track_id, dt=1.0):
        x_min, x_max, y_min, y_max, area = detection
        self.kf = SimpleKalman(x_min, x_max, y_min, y_max, dt=dt)
        self.area = area
        self.id = track_id
        self.missing = 0
        self.history = []

    def predict(self):
        pred = self.kf.predict()
        # >>> DEBUG PRINT
        print(f"[PREDICT] Track {self.id}: predicted centroid = {pred}")
        self.history.append(pred)
        return pred


    def update(self, detection):
        x_min, x_max, y_min, y_max, area = detection
        cx = int(np.mean([x_min, x_max]))
        cy = int(np.mean([y_min, y_max]))
        # >>> DEBUG PRINT
        print(f"[UPDATE] Track {self.id}: updating with detection centroid=({cx},{cy}), area={area}")
        self.kf.update([cx, cy])
        self.area = area
        self.missing = 0


# ============================================================
# FUNCTION: compute_cost
# ============================================================
def compute_cost(tracks, detections, alpha=1.0, beta=0.2):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    cost = np.zeros((len(tracks), len(detections)), dtype=float)
    for i, track in enumerate(tracks):
        pred = track.kf.x[:2].ravel()
        for j, det in enumerate(detections):
            x_min, x_max, y_min, y_max, area = det
            x_new = int(np.mean([x_min, x_max]))
            y_new = int(np.mean([y_min, y_max]))
            dist = np.linalg.norm(pred - np.array([x_new, y_new]))
            area_ratio = abs(np.log((track.area + 1e-3) / (area + 1e-3)))
            cost[i, j] = alpha * dist + beta * area_ratio

            # >>> DEBUG PRINT
            print(f"[COST] Track {track.id} vs Det {j}: pred={pred}, det=({x_new},{y_new}), "
                  f"dist={dist:.2f}, area_ratio={area_ratio:.2f}, cost={cost[i,j]:.2f}")

    return cost


# ============================================================
# FUNCTION: track_particles_from_dataframe
# ============================================================
def track_particles_from_dataframe(df, max_distance=100, max_missing=3, area_threshold=None, dt=1.0):
    """
    df expected columns: ['frame', 'x', 'xx', 'y', 'yy', 'area'] where
    x = xmin, xx = xmax, y = ymin, yy = ymax.
    """
    # Copia y limpiamos
    df = df.copy()

    # Si no tienen columna 'area', la calculamos como bbox area
    if 'area' not in df.columns:
        df['area'] = (df['xx'] - df['x']).abs() * (df['yy'] - df['y']).abs()

    # Opcional: filtrar por area si se pasa threshold
    if area_threshold is None:
        area_threshold = 0  # no filtrar por defecto
    df = df[df['area'] > area_threshold].copy()

    # Aseguramos orden por frame
    df = df.sort_values(['frame']).reset_index(drop=False)  # keep original index in 'index' column
    original_index = df.index  # we'll refer to df.index for label assignment

    tracks = []
    next_id = 0

    # Itera por cada frame en orden
    for frame_idx in sorted(df['frame'].unique()):
        frame_mask = df['frame'] == frame_idx
        frame_detections = df[frame_mask].copy()
        # detections como array de tuplas (xmin,xmax,ymin,ymax,area)
        detections = frame_detections[['x', 'xx', 'y', 'yy', 'area']].to_numpy(dtype=float)
        n_det = len(detections)

        # Predicción para tracks existentes
        for t in tracks:
            t.predict()

        # Si no hay tracks, crear uno por cada detección
        if len(tracks) == 0:
            for k in range(n_det):
                det = tuple(detections[k])
                tracks.append(ParticleTrack(det, next_id, dt=dt))
                # asignar particle usando el índice del dataframe original
                idx = frame_detections.index[k]
                df.at[idx, 'particle_id'] = next_id
                next_id += 1
            continue

        # Si no hay detecciones, solo incrementar missing y limpiar
        if n_det == 0:
            for t in tracks:
                t.missing += 1
            tracks = [t for t in tracks if t.missing < max_missing]
            continue

        # Calcular matriz de costos
        cost = compute_cost(tracks, detections)
        # Asignación con Hungarian
        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for i, j in zip(row_idx, col_idx):
            c = cost[i, j]

            # centroid detection
            x_min, x_max, y_min, y_max, area = detections[j]
            cx = int(np.mean([x_min, x_max]))
            cy = int(np.mean([y_min, y_max]))

            # predicted centroid
            pred = tracks[i].kf.x[:2].ravel()
            dist = np.linalg.norm(pred - np.array([cx, cy]))

            # >>> DEBUG PRINT
            print(f"[ASSIGN] Track {tracks[i].id} → Detection {j}: "
                f"pred={pred}, det=({cx},{cy}), dist={dist:.2f}, cost={c:.2f}, maxdist={max_distance}")

            if c < max_distance:
                print(f"   ✔ ACCEPTED")
                tracks[i].update(tuple(detections[j]))
                idx = frame_detections.index[j]
                df.at[idx, 'particle_id'] = tracks[i].id
                assigned_tracks.add(i)
                assigned_dets.add(j)
            else:
                print(f"   ✘ REJECTED (cost too high)")
                continue


        # Incrementar missing para tracks no asignados
        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.missing += 1

        # Crear nuevos tracks para detecciones no asignadas
        for j in range(n_det):
            if j not in assigned_dets:
                det = tuple(detections[j])
                tracks.append(ParticleTrack(det, next_id, dt=dt))
                idx = frame_detections.index[j]
                df.at[idx, 'particle_id'] = next_id
                next_id += 1

        # Eliminar tracks perdidos
        tracks = [t for t in tracks if t.missing < max_missing]

    # Devolver df con columna 'particle' (si no existía, la creamos)
    if 'particle_id' not in df.columns:
        df['particle_id'] = -1
    # reindex to original ordering (optional)
    return df

# ============================================================
# Example usage (quita o adapta rutas)
# ============================================================
if __name__ == "__main__":
    # lee tu csv (ajusta ruta)

    output_path = "/mnt/CFElab/Data_analysis/Sinker/Shadowgraph_(40297765)/Test/small/results/frames.csv"
    df = pd.read_csv(output_path)
    # Asegúrate que df tenga columnas: ['frame','x','xx','y','yy'] (y opcionalmente 'area')
    df_tracked = track_particles_from_dataframe(df, max_distance=250, max_missing=3, area_threshold=300, dt=1.0)

    image_path = "/mnt/CFElab/Data_analysis/Sinker/Shadowgraph_(40297765)/Test/small/"
    df_tracked.to_csv(output_path, index=False)
    print("Tracking finished. Saved to sinker_tracked.csv")


    save = True
    
    # Sorting the particles
    df_tracked = df_tracked.sort_values(["particle_id", "frame"])

    for particle in df_tracked.particle_id.unique():
        sub = df_tracked[df_tracked.particle_id == particle].sort_values("frame")

        dx = np.diff(sub.x.values)
        dy = np.diff(sub.y.values)
        speed = np.sqrt(dx**2 + dy**2)

        # Assign back into main df (shift by 1 frame)
        idx = sub.index

        df_tracked.loc[idx[:-1], "dx"] = dx
        df_tracked.loc[idx[:-1], "dy"] = dy
        df_tracked.loc[idx[:-1], "speed"] = speed

    # Save final CSV with tracks + velocities
    df_tracked.to_csv(output_path, index=False)

    print("Tracking + velocities complete. Saved to sinker_tracked.csv")

    if save:
        base_path = "/mnt/CFElab/Data_analysis/Sinker/Shadowgraph_(40297765)/Test/small/"
        if not os.path.exists(image_path):
            logger.error(f"Image path not found: {image_path}")
            exit()

        rawframes = pims.ImageSequence(os.path.join(image_path, "*.jpeg"))
        output_dir = os.path.join(base_path, 'tracks')
        os.makedirs(output_dir, exist_ok=True)

        for i in range(len(rawframes)):
            logger.info(f"\n--- Frame {i} ---")
            d = df_tracked[(df_tracked.frame == i) & (df_tracked.area > 300)]
            logger.info(f"rawframe shape: {rawframes[i].shape}")
            logger.info(f"Particles in this frame: {len(d)}")
            logger.info(f"dx stats: min={d.dx.min() if len(d)>0 else None}, max={d.dx.max() if len(d)>0 else None}")

            
            fig, ax = plt.subplots()
            plt.imshow(rawframes[i], cmap='gray', vmin=0, vmax=255)
            colormap = cm.viridis
            colors = 24.0 * 3600 * 10 * np.sqrt(d.dx**2 + d.dy**2) / 1_000_000  # m/day
            norm = Normalize(vmin=0, vmax=255)
            plt.quiver(d.x, d.y, d.dx, -d.dy, color=colormap(norm(colors)), scale_units='xy', scale=0.1, pivot='tail', width=0.0008, headwidth=5, headlength=5)
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
            cbar.set_label('Estimated speed (m/d)')
            plt.savefig(os.path.join(output_dir, f"viz_particles_frame_{i:06d}.png"), dpi=300)
            plt.close(fig)
