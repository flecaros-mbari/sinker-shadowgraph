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

# ============================================================
# CLASS: SimpleKalman
# ============================================================
class SimpleKalman:
    def __init__(self, x_min, x_max, y_min, y_max, dt=1.0, init_velocity=(0.0, 0.0)):
        self.dt = dt
        cx = float(np.mean([x_min, x_max]))
        cy = float(np.mean([y_min, y_max]))
        self.x = np.array([[cx], [cy], [float(init_velocity[0])], [float(init_velocity[1])]], dtype=float)
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.P = np.eye(4) * 500.0
        self.Q = np.diag([1.0, 1.0, 5.0, 5.0])
        self.R = np.eye(2) * 25.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].ravel()

    def update(self, z):
        z = np.array(z, dtype=float).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].ravel()

    @property
    def velocity(self):
        return self.x[2:4].ravel()


# ============================================================
# CLASS: ParticleTrack
# ============================================================
class ParticleTrack:
    AREA_HISTORY_LEN = 7

    def __init__(self, detection, track_id, dt=1.0, is_edge_clipped=False,
                init_velocity=(0.0, 0.0)):
        x_min, x_max, y_min, y_max, area = detection
        self.kf = SimpleKalman(x_min, x_max, y_min, y_max, dt=dt, init_velocity=init_velocity)
        self.area = area
        self.area_history  = [area]
        # A particle spawning at the frame edge is only partially visible, so
        # its first bounding box is smaller than its true size and grows over
        # the next frame or two as more of it enters. area_confirmed==False
        # flags that this track's area estimate is still provisional — used
        # by stitch_edge_fragments() to reattach a short edge-born track to
        # its real, fully-visible continuation after the fact.
        self.area_confirmed = not is_edge_clipped
        self.id = track_id
        self.track_uuid = str(uuid.uuid4())
        self.missing = 0
        self.history = []
        self.age = 0
        self.calib_logged = False

    @property
    def robust_area(self):
        return float(np.median(self.area_history)) if self.area_history else float(self.area)

    def predict(self):
        pred = self.kf.predict()
        self.history.append(pred.copy())
        self.age += 1
        return pred

    def update(self, detection, is_edge_clipped=False):
        x_min, x_max, y_min, y_max, area = detection
        cx = float(np.mean([x_min, x_max]))
        cy = float(np.mean([y_min, y_max]))
        if not is_edge_clipped:
            self.kf.update([cx, cy])
            self.area_history.append(area)
            if len(self.area_history) > self.AREA_HISTORY_LEN:
                self.area_history.pop(0)
            self.area_confirmed = True
        self.area = area
        self.missing = 0


# ============================================================
# HELPERS
# ============================================================
def _is_edge_clipped(det, img_w=4600, img_h=4000, margin=5):
    x_min, x_max, y_min, y_max, _ = det
    return (x_min <= margin or y_min <= margin or
            x_max >= img_w - margin or y_max >= img_h - margin)


def _is_top_clipped(det, img_h=4000, margin=5):
    """
    True only for a particle still entering from the TOP edge — not one
    exiting at the bottom, or one that happens to sit at a left/right edge.
    On this sensor a particle only moves vertically (sinks straight down),
    so top-clipping is the one case that resolves itself: the box keeps
    growing as more of the particle enters, until it's fully confirmed.
    Bottom-clipping never resolves — the particle is leaving for good, so
    there is no "successor" to look for. Left/right clipping (if a particle
    ever spawns at the very side) never resolves either, since x doesn't
    change over the particle's life — it isn't a growing/entering case.
    """
    _x_min, _x_max, y_min, y_max, _ = det
    return y_min <= margin and y_max < img_h - margin


# ============================================================
# FUNCTION: compute_cost
# ============================================================
def compute_cost(tracks, detections, alpha, beta, gamma, img_w, img_h,
                 max_area_ratio=1.3):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    # A real particle's area is stable (~<1% frame-to-frame noise) except
    # when clipped by the image edge, which robust_area already ignores.
    # 10.0 let a track match a detection up to 10x larger/smaller — on this
    # sensor two DIFFERENT particles are far more likely to be that far
    # apart in size than one particle is to change size that much, so a
    # loose ratio here mostly just gives the optimizer license to steal a
    # nearby track's rightful match. 1.3 (30% tolerance) still comfortably
    # absorbs measurement noise while actually discriminating between
    # differently sized particles.
    MAX_AREA_RATIO        = max_area_ratio
    DIST_CAP              = 0.9 * np.sqrt(img_w**2 + img_h**2)
    MIN_AGE_FOR_DIRECTION = 3
    MIN_SPEED_PX          = 10.0

    preds   = np.array([t.kf.x[:2].ravel() for t in tracks])
    areas_t = np.array([t.robust_area      for t in tracks])[:, None]
    vels    = np.array([t.kf.velocity      for t in tracks])
    ages    = np.array([t.age              for t in tracks])

    det_arr = np.asarray(detections, dtype=float)
    centers = np.stack([(det_arr[:, 0] + det_arr[:, 1]) / 2,
                         (det_arr[:, 2] + det_arr[:, 3]) / 2], axis=1)
    areas_d = det_arr[:, 4][None, :]

    diff = preds[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.where(dist > DIST_CAP, np.inf, dist)

    raw_ratio  = np.abs(np.log((areas_t + 1e-3) / (areas_d + 1e-3)))
    norm_denom = np.log(MAX_AREA_RATIO + 1e-3)
    area_term  = np.clip(raw_ratio / norm_denom, 0.0, 1.0)
    area_term  = np.where(raw_ratio > norm_denom, np.inf, area_term)

    cur_pos   = np.array([t.kf.x[:2].ravel() for t in tracks])
    move_vec  = centers[None, :, :] - cur_pos[:, None, :]
    move_len  = np.linalg.norm(move_vec, axis=2, keepdims=True) + 1e-9
    move_unit = move_vec / move_len
    vel_len   = np.linalg.norm(vels, axis=1, keepdims=True) + 1e-9
    vel_unit  = vels / vel_len
    cos_sim   = np.einsum('ti,tdi->td', vel_unit, move_unit)
    dir_cost  = 1.0 - cos_sim

    speed    = vel_len.ravel()
    use_dir  = (ages >= MIN_AGE_FOR_DIRECTION) & (speed >= MIN_SPEED_PX)
    dir_cost = np.where(use_dir[:, None], dir_cost, 0.0)

    return alpha * dist + beta * area_term + gamma * dir_cost


# ============================================================
# FUNCTION: stitch_edge_fragments
# ============================================================
def stitch_edge_fragments(df, img_w, img_h, dt,
                          orphan_max_len=3, max_gap_frames=3,
                          position_tol=150.0, confirm_min_len=8,
                          speed_margin=1.6, fallback_tol=250.0,
                          max_rounds=3):
    """
    Merge short tracks that die right at a frame edge into whichever track
    picks up shortly after in a consistent spot.

    Why this exists: a particle spawning at the image edge is only partially
    visible at first, so its bounding-box area is a fraction of its true
    size and grows every frame as more of it enters. The live tracker's
    area-ratio gate (rightly) rejects large area jumps to tell different
    particles apart — but that means it also rejects the *correct* link
    between an edge-spawned particle's first, small, clipped sighting and
    its second, much bigger, fully-visible one, killing the track after a
    single frame. Loosening that gate live was tried and made things worse:
    the assignment is a joint optimization (linear_sum_assignment) over every
    track and detection in the frame at once, so relaxing the gate for one
    track's row changes which pairs are cheapest for every OTHER track too,
    and lets a completely unrelated far-away track opportunistically match a
    nearby clipped detection meant for someone else — trading a lot of
    cross-particle identity mixing for a little continuity.
    This pass runs after tracking is done, purely on the finished output, one
    fragment at a time — no joint optimization, so it can't cause that kind
    of collateral damage elsewhere.

    A velocity-less (too-short-to-know-its-own-speed) orphan still needs a
    search radius, and a single flat "generous" radius re-creates the same
    problem: it has to be wide enough for a genuinely fast particle, which
    then also lets a slow, barely-moving orphan wrongly latch onto an
    unrelated fast one nearby. Instead we calibrate the radius from the
    orphans' own area: particles on this sensor sink at a roughly constant,
    size-dependent rate, so already-confirmed long tracks give us an
    empirical area -> speed curve, and we look up the orphan's own
    (admittedly partial/under-sized, but still informative) area against it
    to get a speed estimate scaled to *this* particle, not the whole scene's
    worst case.

    Parameters
    ----------
    orphan_max_len : only tracks this short or shorter, AND whose last
        sighting is still entering from the top edge (not exiting at the
        bottom, which never resolves — see _is_top_clipped), are considered
        "possibly cut off at birth" rather than a track that simply lived
        out its whole life and left normally.
    max_gap_frames  : how many frames may separate the orphan's last sighting
        from a candidate successor's first one.
    position_tol    : accepted position residual (px) when the orphan has
        enough points to estimate its own velocity — tight, because that
        extrapolation should be accurate for a real continuation.
    confirm_min_len : tracks at least this long are trusted as calibration
        data for the area -> speed lookup.
    speed_margin    : multiplier on the calibrated speed estimate to leave
        room for noise without falling back to a scene-wide worst case.
    fallback_tol    : radius (px per elapsed frame) used only when no
        calibration data exists yet (e.g. very start of the sequence).
    """
    df = df.copy()
    valid = df.dropna(subset=["track_id"])
    if valid.empty:
        return df

    for _ in range(max_rounds):
        tracks_by_id = {tid: g.sort_values("timestamp")
                        for tid, g in valid.groupby("track_id")}

        summaries = []
        for tid, g in tracks_by_id.items():
            first, last = g.iloc[0], g.iloc[-1]
            vel = None
            if len(g) >= 2:
                dt_span = (last["timestamp"] - first["timestamp"]).total_seconds()
                if dt_span > 0:
                    vel = np.array([last["x"] - first["x"], last["y"] - first["y"]]) / dt_span
            summaries.append(dict(
                tid=tid, n=len(g),
                first_ts=first["timestamp"], last_ts=last["timestamp"],
                last_pos=np.array([float(last["x"]), float(last["y"])]),
                first_pos=np.array([float(first["x"]), float(first["y"])]),
                entering_top=_is_top_clipped(
                    (last["x"], last["xx"], last["y"], last["yy"], 0),
                    img_h=img_h),
                last_area=float(last["area"]), first_area=float(first["area"]),
                max_area=float(g["area"].max()),
                vel=vel,
            ))

        # Area -> speed calibration from tracks long enough to trust, using
        # each one's biggest (least-clipped, closest to true size) area
        # reading and its overall average speed.
        calib = [(s["max_area"], np.linalg.norm(s["vel"]))
                for s in summaries if s["n"] >= confirm_min_len and s["vel"] is not None]
        calib_area  = np.array([c[0] for c in calib])
        calib_speed = np.array([c[1] for c in calib])

        def expected_speed(area):
            if len(calib_area) == 0:
                return None
            nearest = np.argmin(np.abs(np.log(calib_area + 1e-3) - np.log(area + 1e-3)))
            return calib_speed[nearest]

        orphans = [s for s in summaries if s["n"] <= orphan_max_len and s["entering_top"]]
        if not orphans:
            break
        orphans.sort(key=lambda s: s["last_ts"])

        merges = {}
        claimed_successors = set()
        for o in orphans:
            spd = expected_speed(o["max_area"])
            best, best_cost = None, None
            for s in summaries:
                if s["tid"] == o["tid"] or s["tid"] in claimed_successors:
                    continue
                gap_s = (s["first_ts"] - o["last_ts"]).total_seconds()
                if gap_s <= 0 or dt <= 0 or gap_s > max_gap_frames * dt:
                    continue
                if o["vel"] is not None:
                    pred = o["last_pos"] + o["vel"] * gap_s
                    tol = position_tol
                else:
                    pred = o["last_pos"]
                    tol = (spd * speed_margin * gap_s) if spd is not None else fallback_tol
                resid = np.linalg.norm(s["first_pos"] - pred)
                if resid > tol:
                    continue
                # Clipping only shrinks apparent area — a real continuation
                # should look the same size or bigger at the join, not
                # smaller.
                if s["first_area"] < 0.8 * o["last_area"]:
                    continue
                if best is None or resid < best_cost:
                    best, best_cost = s, resid
            if best is not None:
                merges[o["tid"]] = best["tid"]
                claimed_successors.add(best["tid"])

        if not merges:
            break

        for orphan_tid, successor_tid in merges.items():
            valid.loc[valid["track_id"] == successor_tid, "track_id"] = orphan_tid

    df.loc[valid.index, "track_id"] = valid["track_id"]
    return df


# ============================================================
# FUNCTION: track_particles_from_dataframe
# ============================================================
def track_particles_from_dataframe(df, max_distance, max_missing,
                                   area_threshold, dt,
                                   alpha, beta, gamma,
                                   img_w, img_h,
                                   first_track_id=0,
                                   max_area_ratio=1.3):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if 'track_id' not in df.columns:
        df['track_id'] = pd.NA
    df['track_id'] = df['track_id'].astype(object)

    if 'area' not in df.columns:
        df['area'] = (df['xx'] - df['x']).abs() * (df['yy'] - df['y']).abs()

    df = df[df['area'] > area_threshold].copy()
    df = df.sort_values(['timestamp']).reset_index(drop=True)

    unique_timestamps = (
        df[["image_path", "timestamp"]]
        .drop_duplicates(subset=["image_path"])
        .sort_values("timestamp")
    )

    tracks           = []
    next_id          = first_track_id
    prev_timestamp   = None
    track_id_updates = {}

    # Online area -> speed calibration: on this sensor a particle's sinking
    # speed is essentially determined by its (constant, size-dependent) rate,
    # so once a track has run long enough to trust its own velocity estimate,
    # its (size, speed) becomes a reference point for seeding brand-new
    # tracks' initial Kalman velocity — instead of the default zero, which
    # takes several frames to converge and, in the meantime, makes a fresh
    # track's very first prediction untrustworthy (worst for the fastest
    # particles, which move the most before the filter catches up).
    CALIB_MIN_LEN = 5
    calib_area, calib_speed = [], []

    def expected_velocity(area):
        if not calib_area:
            return (0.0, 0.0)
        nearest = int(np.argmin(np.abs(np.log(np.array(calib_area) + 1e-3) - np.log(area + 1e-3))))
        # Particles here sink straight down; only the magnitude is estimated.
        return (0.0, calib_speed[nearest])

    for _, ts_row in unique_timestamps.iterrows():
        current_ts = ts_row["timestamp"]
        img_path   = ts_row["image_path"]

        actual_dt = (
            (current_ts - prev_timestamp).total_seconds()
            if prev_timestamp is not None else dt
        )

        for t in tracks:
            t.kf.dt      = actual_dt
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
                clipped0 = _is_edge_clipped(tuple(detections[k]), img_w=img_w, img_h=img_h)
                nt = ParticleTrack(tuple(detections[k]), next_id, dt=actual_dt,
                                   is_edge_clipped=clipped0,
                                   init_velocity=(0.0, 0.0) if clipped0
                                                 else expected_velocity(detections[k][4]))
                tracks.append(nt)
                track_id_updates[frame_detections.index[k]] = nt.track_uuid
                next_id += 1
            prev_timestamp = current_ts
            continue

        if n_det == 0:
            for t in tracks:
                t.missing += 1
            tracks = [t for t in tracks if t.missing < max_missing]
            prev_timestamp = current_ts
            continue

        cost = compute_cost(tracks, detections,
                            alpha=alpha, beta=beta, gamma=gamma,
                            img_w=img_w, img_h=img_h,
                            max_area_ratio=max_area_ratio)

        # Gate infeasible pairs (over max_distance) to a large sentinel BEFORE
        # optimizing, not after. linear_sum_assignment minimizes the total sum
        # over every finite entry, so an over-threshold-but-finite cost is still
        # "attractive" to the optimizer relative to leaving a track unmatched.
        # That lets a far-away track steal a detection that rightfully belongs
        # to a much closer track, which then gets shoved onto some other bad
        # match — a real match destroyed to slightly lower a doomed one's cost.
        # Clamping every infeasible pair to the same sentinel first makes them
        # all equally unattractive, so the optimizer only economizes over pairs
        # that could actually be accepted.
        row_idx, col_idx = linear_sum_assignment(
            np.where((cost >= max_distance) | np.isinf(cost), 1e18, cost)
        )

        assigned_tracks, assigned_dets = set(), set()

        for i, j in zip(row_idx, col_idx):
            if np.isfinite(cost[i, j]) and cost[i, j] < max_distance:
                clipped = _is_edge_clipped(tuple(detections[j]), img_w=img_w, img_h=img_h)
                tracks[i].update(tuple(detections[j]), is_edge_clipped=clipped)
                if not tracks[i].calib_logged and len(tracks[i].area_history) >= CALIB_MIN_LEN:
                    calib_area.append(tracks[i].robust_area)
                    calib_speed.append(float(np.linalg.norm(tracks[i].kf.velocity)))
                    tracks[i].calib_logged = True
                track_id_updates[frame_detections.index[j]] = tracks[i].track_uuid
                assigned_tracks.add(i)
                assigned_dets.add(j)

        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.missing += 1

        for j in range(n_det):
            if j not in assigned_dets:
                clipped0 = _is_edge_clipped(tuple(detections[j]), img_w=img_w, img_h=img_h)
                nt = ParticleTrack(tuple(detections[j]), next_id, dt=actual_dt,
                                   is_edge_clipped=clipped0,
                                   init_velocity=(0.0, 0.0) if clipped0
                                                 else expected_velocity(detections[j][4]))
                tracks.append(nt)
                track_id_updates[frame_detections.index[j]] = nt.track_uuid
                next_id += 1

        tracks = [t for t in tracks if t.missing < max_missing]
        prev_timestamp = current_ts

    if track_id_updates:
        df.loc[list(track_id_updates.keys()), 'track_id'] = list(track_id_updates.values())

    df = stitch_edge_fragments(df, img_w=img_w, img_h=img_h, dt=dt)

    return df, next_id


# ============================================================
# FUNCTION: compute_velocities
# ============================================================
def compute_velocities(df_tracked):
    df_tracked = df_tracked.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_tracked["timestamp"]):
        df_tracked["timestamp"] = pd.to_datetime(df_tracked["timestamp"])

    df_tracked = df_tracked.sort_values(["track_id", "timestamp"])
    dx_vals, dy_vals, speed_vals = {}, {}, {}

    for particle, sub in df_tracked.groupby("track_id"):
        sub = sub.sort_values("timestamp")
        if len(sub) < 2:
            continue
        dx_px  = np.diff(sub["x"].values)
        dy_px  = np.diff(sub["y"].values)
        dt_sec = np.diff(sub["timestamp"].values).astype("timedelta64[ms]").astype(float) / 1000.0
        dt_sec = np.where(dt_sec == 0, np.nan, dt_sec)
        speed  = np.sqrt((dx_px / dt_sec)**2 + (dy_px / dt_sec)**2)
        idx = sub.index[:-1]
        for k, i in enumerate(idx):
            dx_vals[i]    = dx_px[k]
            dy_vals[i]    = dy_px[k]
            speed_vals[i] = speed[k]

    df_tracked["dx"]    = np.nan
    df_tracked["dy"]    = np.nan
    df_tracked["speed"] = np.nan

    if dx_vals:
        df_tracked.loc[list(dx_vals.keys()),    "dx"]    = list(dx_vals.values())
        df_tracked.loc[list(dy_vals.keys()),    "dy"]    = list(dy_vals.values())
        df_tracked.loc[list(speed_vals.keys()), "speed"] = list(speed_vals.values())

    return df_tracked


# ============================================================
# FUNCTION: smooth_track_velocities
# ============================================================
def smooth_track_velocities(df_tracked, min_len=3):
    """
    Replace compute_velocities()'s frame-to-frame dx/dy/speed with one
    constant (vx, vy) per track, fit robustly across the whole track.

    A real particle here sinks at a constant rate — its true speed does not
    change frame to frame. But raw consecutive-frame differencing measures
    that constant signal plus whatever bounding-box detection jitter is on
    top of it, so even a *perfectly* identity-tracked particle shows visibly
    noisy instantaneous speed. Fitting one slope across the whole track
    instead of differencing point-to-point averages that jitter out, and
    additionally reporting the SAME speed for every row of a track is the
    direct, literal way to enforce "same particle -> same speed".

    Theil-Sen (median of all pairwise slopes) rather than ordinary
    least-squares specifically because it stays robust if a track still has
    a handful of misassigned points in it (a leftover identity mix-up) — a
    few bad points shift an OLS fit noticeably but barely move a median.

    Tracks shorter than `min_len` keep NaN — not enough points for a slope
    estimate to mean anything.
    """
    from scipy.stats import theilslopes

    df_tracked = df_tracked.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_tracked["timestamp"]):
        df_tracked["timestamp"] = pd.to_datetime(df_tracked["timestamp"])

    df_tracked["dx"]    = np.nan
    df_tracked["dy"]    = np.nan
    df_tracked["speed"] = np.nan

    dx_vals, dy_vals, speed_vals = {}, {}, {}

    for tid, sub in df_tracked.groupby("track_id"):
        sub = sub.sort_values("timestamp")
        if len(sub) < min_len:
            continue
        t_sec = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds().values
        if np.ptp(t_sec) <= 0:
            continue
        vx = theilslopes(sub["x"].values, t_sec)[0]
        vy = theilslopes(sub["y"].values, t_sec)[0]
        speed = float(np.hypot(vx, vy))
        for i in sub.index:
            dx_vals[i]    = vx
            dy_vals[i]    = vy
            speed_vals[i] = speed

    if dx_vals:
        df_tracked.loc[list(dx_vals.keys()),    "dx"]    = list(dx_vals.values())
        df_tracked.loc[list(dy_vals.keys()),    "dy"]    = list(dy_vals.values())
        df_tracked.loc[list(speed_vals.keys()), "speed"] = list(speed_vals.values())

    return df_tracked


# ============================================================
# Viridis BGR lookup + colorbar
# ============================================================
_viridis_bgr = (cm.viridis(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]

def _speed_to_bgr(speed_m_day, vmax):
    idx = int(np.clip(speed_m_day / vmax, 0.0, 1.0) * 255) if vmax > 0 else 0
    b, g, r = _viridis_bgr[idx]
    return (int(b), int(g), int(r))

def _draw_colorbar(img, vmax, label="Speed (m/d)"):
    h, w  = img.shape[:2]
    bar_w = 25; pad = 65
    canvas = np.zeros((h, w + bar_w + pad, 3), dtype=np.uint8)
    canvas[:, :w] = img
    for row in range(h):
        frac = 1.0 - row / max(h - 1, 1)
        canvas[row, w: w + bar_w] = _viridis_bgr[int(frac * 255)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k in range(5):
        frac  = k / 4.0
        y_pos = int((1.0 - frac) * (h - 1))
        cv2.line(canvas, (w + bar_w, y_pos), (w + bar_w + 4, y_pos), (210,210,210), 1)
        cv2.putText(canvas, f"{frac*vmax:.3f}", (w + bar_w + 6, y_pos + 4),
                    font, 0.32, (210,210,210), 1, cv2.LINE_AA)
    lbl_img = np.zeros((12, len(label)*8+4, 3), dtype=np.uint8)
    cv2.putText(lbl_img, label, (2,10), font, 0.38, (210,210,210), 1, cv2.LINE_AA)
    lbl_rot = cv2.rotate(lbl_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lh, lw  = lbl_rot.shape[:2]
    y0 = max((h - lh)//2, 0); x0 = w + bar_w + pad - lw - 2
    canvas[y0: y0+lh, x0: x0+lw] = lbl_rot[:min(lh, h-y0)]
    return canvas


# ============================================================
# FUNCTION: visualise_frames_cv2
# ============================================================
def visualise_frames_cv2(df_tracked, output_dir, max_frames=500,
                          min_area=100, metres_per_pixel=10e-6, arrow_scale=5.0,
                          max_arrow_px=150.0):
    os.makedirs(output_dir, exist_ok=True)
    unique_frames = (
        df_tracked[["image_path","timestamp"]].drop_duplicates()
        .sort_values("timestamp").reset_index(drop=True).head(max_frames)
    )
    speed_all = df_tracked["speed"].fillna(0) * metres_per_pixel * 86400
    vmax = max(float(np.nanquantile(speed_all, 0.99)), 1e-9)
    logger.info(f"Rendering {len(unique_frames)} frames -> {output_dir}")
    grouped = {
        ip: grp[(grp["area"] > min_area) & grp["dx"].notna() & grp["dy"].notna()].copy()
        for ip, grp in df_tracked.groupby("image_path")
    }
    for frame_i, frow in tqdm(unique_frames.iterrows(), total=len(unique_frames), desc="cv2 render"):
        img_path  = frow["image_path"]
        save_path = os.path.join(output_dir, f"viz_particles_frame_{frame_i:06d}.png")
        if not os.path.exists(img_path):
            logger.error(f"Image not found: {img_path}"); continue
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            logger.error(f"Could not read: {img_path}"); continue
        frame = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if (raw.ndim == 2 or raw.shape[2] == 1) else raw.copy()
        d = grouped.get(img_path)
        if d is not None and len(d) > 0:
            for row in d.itertuples(index=False):
                spd = float(row.speed) if not np.isnan(row.speed) else 0.0
                colour = _speed_to_bgr(spd * metres_per_pixel * 86400, vmax)
                cx, cy = int(row.x), int(row.y)
                cv2.circle(frame, (cx, cy), 3, colour, -1, cv2.LINE_AA)
                # Clamp arrow length so a genuinely fast particle (or one
                # frame with an unusually large displacement) doesn't draw
                # a line clear across the image — direction and relative
                # speed (via colour) still show, just at a readable scale.
                ax, ay = float(row.dx) * arrow_scale, float(row.dy) * arrow_scale
                arrow_len = np.hypot(ax, ay)
                if arrow_len > max_arrow_px:
                    ax *= max_arrow_px / arrow_len
                    ay *= max_arrow_px / arrow_len
                cv2.arrowedLine(frame, (cx, cy),
                                (int(cx + ax), int(cy + ay)),
                                colour, 5, cv2.LINE_AA, tipLength=0.35)
        cv2.imwrite(save_path, _draw_colorbar(frame, vmax))
    logger.info(f"Done -> {output_dir}")


# ============================================================
# FUNCTION: make_video
# ============================================================
def make_video(frames_dir, output_path, fps=10):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "viz_particles_frame_*.png")))
    if not frame_paths:
        logger.warning(f"No frames in {frames_dir}"); return
    first = cv2.imread(frame_paths[0])
    if first is None:
        logger.error(f"Can't read {frame_paths[0]}"); return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in tqdm(frame_paths, desc="Writing video"):
        img = cv2.imread(p)
        if img is not None: writer.write(img)
    writer.release()
    logger.info(f"Video -> {output_path}  ({len(frame_paths)} frames @ {fps}fps)")


# ============================================================
# MAIN — edit your parameters here, one value each
# ============================================================
if __name__ == "__main__":

    METRES_PER_PIXEL = 10e-6

    # ── tracking parameters ───────────────────────────────────
    # Tuned against the particles_v3 synthetic ground truth (known
    # slow/medium/fast identity per particle) by maximizing per-track
    # class purity while keeping tracks as long/unfragmented as possible.
    # See MAX_AREA_RATIO below — raising BETA only helps because the area
    # gate was tightened; the two were tuned together.
    ALPHA          = 1.0     # distance weight
    BETA           = 1500.0  # area mismatch weight
    GAMMA          = 200.0   # direction consistency weight
    MAX_DISTANCE   = 2200    # max cost to accept an assignment
    AREA_THRESHOLD = 0       # minimum detection area (px²)
    # Checked directly against raw ground-truth pixel displacement (not the
    # nominal m/day figures, which need a metres-per-pixel conversion that
    # turned out not to be exactly 10um/px for this render): max_missing=2
    # recovered every class's true speed to within 0.6%, while 5 undershot
    # the fast class by ~8%. A larger tolerance bridges more real gaps, but
    # also gives more opportunities to bridge onto a *different*, similarly
    # sized/positioned particle instead of the right one.
    MAX_MISSING    = 2       # frames a track can go unmatched before dying
    MAX_AREA_RATIO = 1.3     # max area(track)/area(detection) ratio to allow a match
    # ─────────────────────────────────────────────────────────

    # single folder name built from the values above — no duplicates
    run_tag = (f"alpha-{ALPHA}_beta-{BETA}_gamma-{GAMMA}"
               f"_dist-{MAX_DISTANCE}_area-{AREA_THRESHOLD}"
               f"_arearatio-{MAX_AREA_RATIO}_missing-{MAX_MISSING}")

    output_paths = [
        "/mbari/Tempbox/fernanda/particles_v3/"
        # "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/"
    ]

    base_path_map = {
        "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/": "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/results/",
        "/mbari/Tempbox/fernanda/particles_v3/": "/mbari/Tempbox/fernanda/particles_v3/results/"
    }

    for output_path in output_paths:

        # NOTE: particles_v3 (and any folder produced by testing-blener.py)
        # holds a per-frame CSV for every single image PLUS one merged
        # "combined.csv" (from csv_combination.py) that has all frames'
        # detections together. Tracking needs the merged file — a per-frame
        # CSV has only one timestamp in it, so running the tracker on one
        # produces a meaningless single-frame "track" per detection. Globbing
        # "*.csv" swept up all 1367 files and fed each through the full
        # tracking + video pipeline as if it were an independent recording.
        combined_csv = os.path.join(output_path, "combined.csv")
        if os.path.exists(combined_csv):
            csv_files = [combined_csv]
        else:
            csv_files = sorted([
                os.path.join(output_path, f)
                for f in os.listdir(output_path) if f.endswith(".csv")
            ])
        logger.info(f"Found {len(csv_files)} CSV files in {output_path}")

        global_next_id = 0

        for csv_path in tqdm(csv_files, desc="Processing CSVs"):

            df_original = pd.read_csv(csv_path)
            df_original["timestamp"] = pd.to_datetime(df_original["timestamp"])

            frame_times = (
                df_original[["image_path", "timestamp"]]
                .drop_duplicates().sort_values("timestamp")
            )
            dt_median = frame_times["timestamp"].diff().dropna().median().total_seconds()
            df_original["frame_rate"] = 1.0 / dt_median

            img_w = int(df_original["image_width"].iloc[0])  if "image_width"  in df_original.columns else 4600
            img_h = int(df_original["image_height"].iloc[0]) if "image_height" in df_original.columns else 4000

            phys_max_dist = min(MAX_DISTANCE * dt_median,
                                0.9 * np.sqrt(img_w**2 + img_h**2))

            df_tracked, global_next_id = track_particles_from_dataframe(
                df_original.copy(),
                max_distance=phys_max_dist,
                max_missing=MAX_MISSING,
                area_threshold=AREA_THRESHOLD,
                dt=dt_median,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
                img_w=img_w,
                img_h=img_h,
                first_track_id=global_next_id,
                max_area_ratio=MAX_AREA_RATIO,
            )
            df_tracked = smooth_track_velocities(df_tracked)

            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            combo_dir = os.path.join(output_path, "param_search", run_tag)
            os.makedirs(combo_dir, exist_ok=True)
            df_tracked.to_csv(os.path.join(combo_dir, f"{base_name}.csv"), index=False)

            output_dir = os.path.join(base_path_map[output_path], run_tag, base_name)
            visualise_frames_cv2(df_tracked, output_dir, max_frames=500,
                                  min_area=0, metres_per_pixel=METRES_PER_PIXEL,
                                  arrow_scale=0.2, max_arrow_px=150.0)
            make_video(output_dir,
                       os.path.join(output_dir, f"{base_name}_video.mp4"),
                       fps=1)

        logger.info(f"Finished {output_path}")