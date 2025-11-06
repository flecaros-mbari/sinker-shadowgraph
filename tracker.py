import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

# ============================================================
# CLASS: SimpleKalman
# ============================================================
class SimpleKalman:
    def __init__(self, x,xx, y,yy, dt=1.0):
        self.dt = dt
        self.x = np.array([[int(np.mean([x,xx]))], [int(np.mean([y,yy]))], [0], [0]])
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        print(f"Predicted state: {self.x.ravel()}")
        return self.x[:2].ravel()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        print(f"Updated state: {self.x.ravel()}")

# ============================================================
# CLASS: ParticleTrack
# ============================================================
class ParticleTrack:
    def __init__(self, detection, track_id):
        x,xx, y,yy, area = detection
        self.kf = SimpleKalman(x,xx, y,yy)
        self.area = area
        self.id = track_id
        self.missing = 0
        self.history = []

    def predict(self):
        pred = self.kf.predict()
        self.history.append(pred)
        return pred

    def update(self, detection):
        x,xx, y,yy, area = detection
        self.kf.update([int(np.mean([x,xx])), int(np.mean([y,yy]))])
        self.area = area
        self.missing = 0

# ============================================================
# FUNCTION: compute_cost
# ============================================================
def compute_cost(tracks, detections, alpha=1.0, beta=0.2):
    cost = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        pred = track.kf.x[:2].ravel()
        for j, det in enumerate(detections):
            x,xx, y,yy, area = det
            x_new, y_new = int(np.mean([x, xx])), int(np.mean([y, yy]))
            dist = np.linalg.norm(pred - np.array([x_new, y_new]))
            area_ratio = abs(np.log((track.area + 1e-3) / (area + 1e-3)))
            cost[i, j] = alpha * dist + beta * area_ratio
    print(f"Cost matrix:\n{cost}")
    return cost

# ============================================================
# FUNCTION: track_particles_from_dataframe
# ============================================================
def track_particles_from_dataframe(df, max_distance=400, max_missing=3):
    tracks = []
    next_id = 0

    df = df[df['area'] > 40].copy()

    # Creamos columna 'particle' vacía
    df['particle'] = -1

    

    # Itera por cada frame
    for frame_idx in sorted(df['frame'].unique()):
        frame_detections = df[df['frame'] == frame_idx].copy()
        detections = frame_detections[['x','xx', 'yy','x', 'area']].to_numpy()
        print(f"\nFrame {frame_idx} — {len(detections)} detections")

        # Predicción
        for t in tracks:
            t.predict()

        # Si no hay tracks, crear nuevos
        if len(tracks) == 0:
            for idx, det in enumerate(detections):
                tracks.append(ParticleTrack(det, next_id))
                df.loc[frame_detections.index[idx], 'particle'] = next_id
                print(f"New track {next_id} created for detection {det}")
                next_id += 1
            continue

        # Calcular costos
        cost = compute_cost(tracks, detections)

        # Asignación Húngara
        row_idx, col_idx = linear_sum_assignment(cost)
        assigned_tracks, assigned_dets = set(), set()

        # Actualizar tracks asignados
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < max_distance:
                tracks[i].update(detections[j])
                df.loc[frame_detections.index[j], 'particle'] = tracks[i].id
                assigned_tracks.add(i)
                assigned_dets.add(j)
                print(f"Track {tracks[i].id} assigned to detection {detections[j]}")

        # Incrementar missing
        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.missing += 1
                print(f"Track {t.id} missing incremented to {t.missing}")

        # Nuevos tracks para detecciones no asignadas
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                tracks.append(ParticleTrack(det, next_id))
                df.loc[frame_detections.index[j], 'particle'] = next_id
                print(f"New track {next_id} created for unmatched detection {det}")
                next_id += 1

        # Eliminar tracks perdidos
        tracks = [t for t in tracks if t.missing < max_missing]

    return df


# ============================================================
# Example of use
# ============================================================
if __name__ == "__main__":
    
    # Simulating the csv
    df = pd.read_csv('/Users/fernandalecaros/Downloads/sinker/results/frames.csv')
    df_tracked = track_particles_from_dataframe(df)
    df_tracked.to_csv("/Users/fernandalecaros/Downloads/sinker/results/sinker_tracked.csv", index=False)
    print("\nTracked DataFrame:")
    print(df_tracked)
