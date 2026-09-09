import numpy as np
import uuid
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
from loguru import logger
import matplotlib.cm as cm
import cv2
from tqdm import tqdm
import glob
from itertools import product

# ============================================================
# CLASS: SimpleKalman
# ============================================================
class SimpleKalman:
    def __init__(self, x_min, x_max, y_min, y_max, dt=1.0):
        self.dt = dt
        cx = int(np.mean([x_min, x_max]))
        cy = int(np.mean([y_min, y_max]))
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
        self.track_uuid = str(uuid.uuid4())
        self.missing = 0
        self.history = []

    def predict(self):
        pred = self.kf.predict()
        self.history.append(pred)
        return pred

    def update(self, detection):
        x_min, x_max, y_min, y_max, area = detection
        cx = int(np.mean([x_min, x_max]))
        cy = int(np.mean([y_min, y_max]))
        self.kf.update([cx, cy])
        self.area = area
        self.missing = 0


# ============================================================
# FUNCTION: compute_cost  [OPTIMIZED: fully vectorized, no Python loops]
# ============================================================
def compute_cost(tracks, detections, alpha=1.0, beta=10):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    # Stack all track predicted positions and areas — no inner loop
    preds   = np.array([t.kf.x[:2].ravel() for t in tracks])          # (T, 2)
    areas_t = np.array([t.area for t in tracks])[:, None]              # (T, 1)

    # Stack all detection centres and areas — no inner loop
    det_arr  = np.asarray(detections)                                   # (D, 5)
    centers  = np.stack([(det_arr[:, 0] + det_arr[:, 1]) / 2,
                          (det_arr[:, 2] + det_arr[:, 3]) / 2], axis=1) # (D, 2)
    areas_d  = det_arr[:, 4][None, :]                                   # (1, D)

    # Broadcast: distance (T, D) and area ratio (T, D)
    dist       = np.linalg.norm(preds[:, None, :] - centers[None, :, :], axis=2)
    area_ratio = np.abs(np.log((areas_t + 1e-3) / (areas_d + 1e-3)))

    return alpha * dist + beta * area_ratio


# ============================================================
# FUNCTION: track_particles_from_dataframe  [OPTIMIZED: batched df writes]
# ============================================================
def track_particles_from_dataframe(df, max_distance=100, max_missing=3,
                                   area_threshold=None, dt=1.0,
                                   alpha=1, beta=1,
                                   first_track_id=0):
    """
    Parameters
    ----------
    first_track_id : int
        Starting value for the track ID counter.

    Returns
    -------
    df_tracked  : pd.DataFrame  — input df with 'track_id' column filled.
    next_track_id : int         — pass to next call to keep IDs globally unique.
    """
    df = df.copy()
    # Parse timestamps once here; compute_velocities receives already-parsed df
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if 'area' not in df.columns:
        df['area'] = (df['xx'] - df['x']).abs() * (df['yy'] - df['y']).abs()

    if area_threshold is None:
        area_threshold = 0
    df = df[df['area'] > area_threshold].copy()

    df = df.sort_values(['timestamp']).reset_index(drop=True)

    unique_timestamps = (
        df[["image_path", "timestamp"]]
        .drop_duplicates(subset=["image_path"])
        .sort_values("timestamp")
    )

    tracks       = []
    next_id      = first_track_id
    prev_timestamp = None

    # OPTIMIZATION: accumulate all track_id writes, flush once at the end
    track_id_updates = {}

    for _, ts_row in unique_timestamps.iterrows():
        current_ts = ts_row["timestamp"]
        img_path   = ts_row["image_path"]

        actual_dt = (current_ts - prev_timestamp).total_seconds() if prev_timestamp is not None else dt

        for t in tracks:
            t.kf.dt    = actual_dt
            t.kf.F[0, 2] = actual_dt
            t.kf.F[1, 3] = actual_dt

        frame_mask       = df["image_path"] == img_path
        frame_detections = df[frame_mask].copy()
        detections       = frame_detections[['x', 'xx', 'y', 'yy', 'area']].to_numpy(dtype=float)
        n_det            = len(detections)

        for t in tracks:
            t.predict()

        if len(tracks) == 0:
            for k in range(n_det):
                new_track = ParticleTrack(tuple(detections[k]), next_id, dt=actual_dt)
                tracks.append(new_track)
                # OPTIMIZATION: stage the write instead of calling df.at[]
                track_id_updates[frame_detections.index[k]] = new_track.track_uuid
                next_id += 1
            prev_timestamp = current_ts
            continue

        if n_det == 0:
            for t in tracks:
                t.missing += 1
            tracks = [t for t in tracks if t.missing < max_missing]
            prev_timestamp = current_ts
            continue

        cost = compute_cost(tracks, detections, alpha=alpha, beta=beta)
        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_tracks, assigned_dets = set(), set()

        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < max_distance:
                tracks[i].update(tuple(detections[j]))
                # OPTIMIZATION: stage the write
                track_id_updates[frame_detections.index[j]] = tracks[i].track_uuid
                assigned_tracks.add(i)
                assigned_dets.add(j)

        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.missing += 1

        for j in range(n_det):
            if j not in assigned_dets:
                new_track = ParticleTrack(tuple(detections[j]), next_id, dt=actual_dt)
                tracks.append(new_track)
                # OPTIMIZATION: stage the write
                track_id_updates[frame_detections.index[j]] = new_track.track_uuid
                next_id += 1

        tracks = [t for t in tracks if t.missing < max_missing]
        prev_timestamp = current_ts

    # OPTIMIZATION: single batched write for all frames
    if track_id_updates:
        df.loc[list(track_id_updates.keys()), 'track_id'] = list(track_id_updates.values())

    return df, next_id


# ============================================================
# FUNCTION: compute_velocities  [OPTIMIZED: single bulk assignment]
# ============================================================
def compute_velocities(df_tracked):
    # Timestamps already parsed upstream — avoid redundant pd.to_datetime call
    df_tracked = df_tracked.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_tracked["timestamp"]):
        df_tracked["timestamp"] = pd.to_datetime(df_tracked["timestamp"])

    df_tracked = df_tracked.sort_values(["track_id", "timestamp"])

    # OPTIMIZATION: build result arrays alongside index lists, assign once
    dx_vals    = {}
    dy_vals    = {}
    speed_vals = {}

    for particle, sub in df_tracked.groupby("track_id"):
        sub = sub.sort_values("timestamp")
        if len(sub) < 2:
            continue

        dx_px  = np.diff(sub["x"].values)
        dy_px  = np.diff(sub["y"].values)
        dt_sec = np.diff(sub["timestamp"].values).astype("timedelta64[ms]").astype(float) / 1000.0
        dt_sec = np.where(dt_sec == 0, np.nan, dt_sec)
        speed  = np.sqrt((dx_px / dt_sec) ** 2 + (dy_px / dt_sec) ** 2)

        idx = sub.index[:-1]   # one fewer than rows (diff shrinks by 1)
        for k, i in enumerate(idx):
            dx_vals[i]    = dx_px[k]
            dy_vals[i]    = dy_px[k]
            speed_vals[i] = speed[k]

    # Single bulk assignment per column
    df_tracked["dx"]    = np.nan
    df_tracked["dy"]    = np.nan
    df_tracked["speed"] = np.nan

    if dx_vals:
        df_tracked.loc[list(dx_vals.keys()),    "dx"]    = list(dx_vals.values())
        df_tracked.loc[list(dy_vals.keys()),    "dy"]    = list(dy_vals.values())
        df_tracked.loc[list(speed_vals.keys()), "speed"] = list(speed_vals.values())

    return df_tracked


# ============================================================
# Viridis BGR lookup table
# ============================================================
_viridis_bgr = (cm.viridis(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]


def _speed_to_bgr(speed_m_day: float, vmax: float) -> tuple:
    idx = int(np.clip(speed_m_day / vmax, 0.0, 1.0) * 255) if vmax > 0 else 0
    b, g, r = _viridis_bgr[idx]
    return (int(b), int(g), int(r))


# ============================================================
# FUNCTION: _draw_colorbar
# ============================================================
def _draw_colorbar(img: np.ndarray, vmax: float,
                   label: str = "Speed (m/d)") -> np.ndarray:
    h, w   = img.shape[:2]
    bar_w  = 25
    pad    = 65
    canvas = np.zeros((h, w + bar_w + pad, 3), dtype=np.uint8)
    canvas[:, :w] = img

    for row in range(h):
        frac = 1.0 - row / max(h - 1, 1)
        canvas[row, w: w + bar_w] = _viridis_bgr[int(frac * 255)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for k in range(5):
        frac  = k / 4.0
        y_pos = int((1.0 - frac) * (h - 1))
        val   = frac * vmax
        cv2.line(canvas, (w + bar_w, y_pos), (w + bar_w + 4, y_pos), (210, 210, 210), 1)
        cv2.putText(canvas, f"{val:.3f}", (w + bar_w + 6, y_pos + 4),
                    font, 0.32, (210, 210, 210), 1, cv2.LINE_AA)

    lbl_img = np.zeros((12, len(label) * 8 + 4, 3), dtype=np.uint8)
    cv2.putText(lbl_img, label, (2, 10), font, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    lbl_rot = cv2.rotate(lbl_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lh, lw  = lbl_rot.shape[:2]
    y0 = max((h - lh) // 2, 0)
    x0 = w + bar_w + pad - lw - 2
    canvas[y0: y0 + lh, x0: x0 + lw] = lbl_rot[:min(lh, h - y0)]

    return canvas


# ============================================================
# FUNCTION: visualise_frames_cv2  [OPTIMIZED: itertuples instead of iterrows]
# ============================================================
def visualise_frames_cv2(df_tracked, output_dir, max_frames=500,
                          min_area=100, metres_per_pixel=10e-6,
                          arrow_scale=5.0):
    os.makedirs(output_dir, exist_ok=True)

    unique_frames = (
        df_tracked[["image_path", "timestamp"]]
        .drop_duplicates()
        .sort_values("timestamp")
        .reset_index(drop=True)
        .head(max_frames)
    )

    speed_all = df_tracked["speed"].fillna(0) * metres_per_pixel * 86400
    vmax = float(np.nanquantile(speed_all, 0.99))
    vmax = max(vmax, 1e-9)

    logger.info(f"Rendering {len(unique_frames)} frames -> {output_dir}  (vmax={vmax:.4f} m/d)")

    # Pre-build a per-image lookup so we don't filter the full df every frame
    grouped = {
        img_path: grp[
            (grp["area"] > min_area) &
            grp["dx"].notna() &
            grp["dy"].notna()
        ].copy()
        for img_path, grp in df_tracked.groupby("image_path")
    }

    for frame_i, frow in tqdm(unique_frames.iterrows(), total=len(unique_frames),
                               desc="cv2 render"):
        img_path  = frow["image_path"]
        save_path = os.path.join(output_dir, f"viz_particles_frame_{frame_i:06d}.png")

        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}")
            continue

        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            logger.error(f"Could not read: {img_path}")
            continue

        if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
            frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            frame = raw.copy()

        d = grouped.get(img_path)
        if d is not None and len(d) > 0:
            # OPTIMIZATION: itertuples ~5–10× faster than iterrows
            for row in d.itertuples(index=False):
                cx  = int(row.x)
                cy  = int(row.y)
                dx  = float(row.dx)
                dy  = float(row.dy)
                spd = float(row.speed) if not np.isnan(row.speed) else 0.0

                speed_m_day = spd * metres_per_pixel * 86400
                colour      = _speed_to_bgr(speed_m_day, vmax)

                tip_x = int(cx + dx * arrow_scale)
                tip_y = int(cy + dy * arrow_scale)

                cv2.circle(frame, (cx, cy), radius=3, color=colour,
                           thickness=-1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y),
                                color=colour, thickness=5,
                                line_type=cv2.LINE_AA, tipLength=0.35)

        frame_out = _draw_colorbar(frame, vmax, label="Speed (m/d)")
        cv2.imwrite(save_path, frame_out)

    logger.info(f"Done -> {output_dir}")


# ============================================================
# FUNCTION: make_video
# ============================================================
def make_video(frames_dir, output_path, fps=10):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "viz_particles_frame_*.png")))
    if not frame_paths:
        logger.warning(f"No frames found in {frames_dir}, skipping video.")
        return

    first = cv2.imread(frame_paths[0])
    if first is None:
        logger.error(f"Could not read first frame: {frame_paths[0]}")
        return

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for p in tqdm(frame_paths, desc="Writing video"):
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)

    writer.release()
    logger.info(f"Video saved -> {output_path}  ({len(frame_paths)} frames @ {fps} fps)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    METRES_PER_PIXEL = 10e-6
    MAX_SPEED_M_DAY  = 1000

    param_grid = {
        "alpha":          [1.0],
        "beta":           [0.2],
        "max_distance":   [1000],
        "area_threshold": [300],
    }

    param_combinations = list(product(
        param_grid["alpha"],
        param_grid["beta"],
        param_grid["max_distance"],
        param_grid["area_threshold"],
    ))

    output_paths = [
        # "/mnt/CFElab/Data_analysis/Sinker/Test/newdetection/"
        "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/"
    ]

    base_path_map = {
        "/mnt/CFElab/Data_analysis/Sinker/test_0.5h/": "/mnt/CFElab/Data_analysis/Sinker/tracking_images_0.5H_v2/",
        "/mnt/CFElab/Data_analysis/Sinker/test_1h/":   "/mnt/CFElab/Data_analysis/Sinker/tracking_images_H_v2/",
        "/mnt/CFElab/Data_analysis/Sinker/test_2h/":   "/mnt/CFElab/Data_analysis/Sinker/tracking_images_2H_v2/",
        "/mnt/CFElab/Data_analysis/Sinker/wednesday copy/" : "/mnt/CFElab/Data_analysis/Sinker/wednesday copy/all/",
        "/mnt/CFElab/Data_analysis/Sinker/Test/newdetection/": "/mnt/CFElab/Data_analysis/Sinker/Test/newdetection_results/",
        "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/":"/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/results/"
    }

    for output_path in output_paths:

        csv_files = sorted([
            os.path.join(output_path, f)
            for f in os.listdir(output_path)
            if f.endswith(".csv")
        ])
        logger.info(f"Found {len(csv_files)} CSV files in {output_path}")

        for alpha, beta, max_dist, area_th in param_combinations:

            global_next_id = 0

            for csv_path in tqdm(csv_files, desc=f"CSV files | a={alpha} b={beta} d={max_dist} ar={area_th}"):

                df_original = pd.read_csv(csv_path)
                df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])

                frame_times = (
                    df_original[["image_path", "timestamp"]]
                    .drop_duplicates()
                    .sort_values("timestamp")
                )
                dt_median = frame_times["timestamp"].diff().dropna().median().total_seconds()
                df_original["frame_rate"] = 1.0 / dt_median

                df = df_original.copy()

                phys_max_dist = (MAX_SPEED_M_DAY / 86400) / METRES_PER_PIXEL * dt_median * (max_dist / 120.0)
                print(phys_max_dist)

                df_tracked, global_next_id = track_particles_from_dataframe(
                    df,
                    max_distance=phys_max_dist,
                    max_missing=3,
                    area_threshold=area_th,
                    dt=dt_median,
                    alpha=alpha,
                    beta=beta,
                    first_track_id=global_next_id,
                )
                df_tracked = compute_velocities(df_tracked)

                base_name  = os.path.splitext(os.path.basename(csv_path))[0]
                combo_name = f"{base_name}_alpha-{alpha}_beta-{beta}_dist-{max_dist}_area-{area_th}.csv"
                combo_dir  = os.path.join(output_path, "param_search")
                os.makedirs(combo_dir, exist_ok=True)
                df_tracked.to_csv(os.path.join(combo_dir, combo_name), index=False)

                # ── per-CSV output folder (now includes base_name) ──────────
                output_dir = os.path.join(
                    base_path_map[output_path],
                    f"tracks_alpha-{alpha}_beta-{beta}_dist-{max_dist}_area-{area_th}",
                    base_name,          # <── one sub-folder per CSV
                )

                # ── visualise & encode video for THIS csv ────────────────────
                visualise_frames_cv2(
                    df_tracked,
                    output_dir,
                    max_frames=500,
                    min_area=300,
                    metres_per_pixel=METRES_PER_PIXEL,
                    arrow_scale=5.0,
                )
                video_path = os.path.join(
                    output_dir,
                    f"video_alpha-{alpha}_beta-{beta}_dist-{max_dist}_area-{area_th}.mp4"
                )
                make_video(output_dir, video_path, fps=1)

        logger.info(f"Finished {output_path}")