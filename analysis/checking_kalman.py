import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracking_imporoved import SimpleKalman, ParticleTrack, compute_cost, _is_edge_clipped
from scipy.optimize import linear_sum_assignment


def run_tracker(csv_path, alpha=1.0, beta=400.0, gamma=200.0,
                max_distance=2000, area_threshold=31416,
                max_missing=3, img_w=4600, img_h=4000):
    """Run the full tracker and return the record dict {uuid: [frame_dicts]}."""

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "area" not in df.columns:
        df["area"] = (df["xx"] - df["x"]).abs() * (df["yy"] - df["y"]).abs()
    df = df[df["area"] > area_threshold].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_frames = (
        df[["image_path", "timestamp"]]
        .drop_duplicates(subset=["image_path"])
        .sort_values("timestamp")
    )

    tracks, next_id, prev_timestamp = [], 0, None
    record = {}

    for frame_i, ts_row in enumerate(unique_frames.itertuples(index=False)):
        current_ts = ts_row.timestamp
        img_path   = ts_row.image_path

        actual_dt = (
            (current_ts - prev_timestamp).total_seconds()
            if prev_timestamp is not None else 1.0
        )

        for t in tracks:
            t.kf.dt = actual_dt
            t.kf.F[0, 2] = actual_dt
            t.kf.F[1, 3] = actual_dt

        frame_mask       = df["image_path"] == img_path
        frame_detections = df[frame_mask].copy()
        detections       = frame_detections[["x", "xx", "y", "yy", "area"]].to_numpy(dtype=float)
        n_det            = len(detections)

        predictions = {}
        for t in tracks:
            pred = t.kf.predict()
            predictions[t.track_uuid] = pred.copy()
            t.age += 1
            t.history.append(pred.copy())

        if len(tracks) == 0:
            for k in range(n_det):
                nt = ParticleTrack(tuple(detections[k]), next_id, dt=actual_dt)
                tracks.append(nt)
                record[nt.track_uuid] = []
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
                            img_w=img_w, img_h=img_h)
        # Gate infeasible pairs before optimizing — see tracking_imporoved.py
        # for why post-hoc thresholding lets Hungarian steal good matches.
        row_idx, col_idx = linear_sum_assignment(
            np.where((cost >= max_distance) | np.isinf(cost), 1e18, cost)
        )

        assigned_tracks, assigned_dets = set(), set()

        for i, j in zip(row_idx, col_idx):
            if np.isfinite(cost[i, j]) and cost[i, j] < max_distance:
                t   = tracks[i]
                det = tuple(detections[j])
                clipped   = _is_edge_clipped(det, img_w=img_w, img_h=img_h)
                actual_cx = (det[0] + det[1]) / 2
                actual_cy = (det[2] + det[3]) / 2
                pred_xy   = predictions[t.track_uuid]

                if t.track_uuid not in record:
                    record[t.track_uuid] = []
                record[t.track_uuid].append({
                    "frame":    frame_i,
                    "pred_x":  pred_xy[0],
                    "pred_y":  pred_xy[1],
                    "actual_x": actual_cx,
                    "actual_y": actual_cy,
                })

                t.update(det, is_edge_clipped=clipped)
                assigned_tracks.add(i)
                assigned_dets.add(j)

        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.missing += 1

        for j in range(n_det):
            if j not in assigned_dets:
                nt = ParticleTrack(tuple(detections[j]), next_id, dt=actual_dt)
                tracks.append(nt)
                record[nt.track_uuid] = []
                next_id += 1

        tracks = [t for t in tracks if t.missing < max_missing]
        prev_timestamp = current_ts

    return record


def plot_particle(uuid, data, idx, total):
    df_diag = pd.DataFrame(data)
    err_x = df_diag["pred_x"] - df_diag["actual_x"]
    err_y = df_diag["pred_y"] - df_diag["actual_y"]
    err   = np.sqrt(err_x**2 + err_y**2)

    print(f"\n{'='*60}")
    print(f"Particle {idx+1}/{total}  |  UUID: {uuid}")
    print(f"Matched frames: {len(df_diag)}")
    print(f"Error (px)  mean={err.mean():.2f}  std={err.std():.2f}  max={err.max():.2f}")
    print(df_diag.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Particle {idx+1}/{total}  —  {uuid[:8]}…  ({len(df_diag)} frames)",
        fontsize=11
    )

    # 1. XY trajectory
    ax = axes[0]
    ax.plot(df_diag["actual_x"], df_diag["actual_y"],
            "o-", color="steelblue", label="Actual", markersize=5, linewidth=1.2)
    ax.plot(df_diag["pred_x"], df_diag["pred_y"],
            "s--", color="tomato", label="Predicted", markersize=5, linewidth=1.2)
    for _, r in df_diag.iterrows():
        ax.annotate("", xy=(r.actual_x, r.actual_y),
                    xytext=(r.pred_x, r.pred_y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))
    # label frame numbers
    for _, r in df_diag.iterrows():
        ax.text(r.actual_x, r.actual_y, str(int(r.frame)),
                fontsize=6, color="steelblue", ha="left", va="bottom")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("XY trajectory"); ax.legend(); ax.invert_yaxis()

    # 2. Per-axis error
    ax = axes[1]
    ax.plot(df_diag["frame"], err_x, label="Δx", color="darkorange")
    ax.plot(df_diag["frame"], err_y, label="Δy", color="mediumpurple")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Frame"); ax.set_ylabel("Error (px)")
    ax.set_title("Per-axis error (pred − actual)"); ax.legend()

    # 3. Euclidean error
    ax = axes[2]
    ax.plot(df_diag["frame"], err, color="crimson", marker="o", markersize=4)
    ax.axhline(err.mean(), color="black", ls="--", lw=0.9,
               label=f"Mean = {err.mean():.1f} px")
    ax.set_xlabel("Frame"); ax.set_ylabel("Euclidean error (px)")
    ax.set_title("Total prediction error"); ax.legend()

    plt.tight_layout()
    plt.show()


def browse_particles(csv_path, min_track_length=3, **tracker_kwargs):
    """
    Step through every track one by one.
    At the prompt type:
      Enter  → next particle
      p      → previous particle
      q      → quit
      <uuid> → jump to that specific track
    """
    print("Running tracker …")
    record = run_tracker(csv_path, **tracker_kwargs)

    # sort by track length descending so the most interesting ones come first
    sorted_uuids = sorted(
        [u for u, d in record.items() if len(d) >= min_track_length],
        key=lambda u: len(record[u]),
        reverse=True
    )
    total = len(sorted_uuids)
    print(f"\nFound {total} tracks with >= {min_track_length} matched frames.")
    print("Controls: Enter=next  p=previous  q=quit  <uuid>=jump\n")

    idx = 0
    while 0 <= idx < total:
        uuid = sorted_uuids[idx]
        plot_particle(uuid, record[uuid], idx, total)

        cmd = input(f"\n[{idx+1}/{total}] Enter / p / q / <uuid>: ").strip().lower()

        if cmd == "q":
            print("Quit.")
            break
        elif cmd == "p":
            idx = max(0, idx - 1)
        elif cmd == "":
            idx += 1
        elif cmd in record:
            idx = sorted_uuids.index(cmd) if cmd in sorted_uuids else idx
        else:
            print(f"  '{cmd}' not recognised — Enter=next  p=prev  q=quit")


# ── run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    browse_particles(
        csv_path         = "/mnt/CFElab/Data_analysis/Sinker/Test/newbackground/sdcat/param_search/alpha-1.0_beta-400.0_gamma-200.0_dist-2000_area-300/20250825T074058Z.csv",
        min_track_length = 3,       # skip very short/spurious tracks
        alpha            = 1.0,
        beta             = 400.0,
        gamma            = 200.0,
        max_distance     = 2000,
        area_threshold   = 300,
        max_missing      = 3,
        img_w            = 4600,
        img_h            = 4000,
    )