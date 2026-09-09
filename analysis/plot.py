import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from glob import glob

# ==============================
# CONSTANTS
# ==============================
PIXEL_SIZE_UM = 10
SECONDS_PER_DAY = 24 * 3600
UM_TO_M = 1e-6

def pix_per_sec_to_m_per_day(value):
    return value * PIXEL_SIZE_UM * UM_TO_M * SECONDS_PER_DAY

def mean_without_outliers(x):
    mu = x.mean()
    sigma = x.std()
    x_filt = x[(x >= mu - 2*sigma) & (x <= mu + 2*sigma)]
    return x_filt.mean()

def is_even_hour(ts, min):
    return ts.hour % 2 == 0 and ts.minute < min


# ==============================================================
# SECTION 1 — quick single-CSV overview plots (legacy plot.py logic)
# ==============================================================
# Recomputes velocity from consecutive-frame (x, y) displacement rather than
# using the tracker's own dx/dy/speed columns, and expects a "particle"
# column (renamed here from "track_id" to match a tracked.csv straight out
# of tracking_improved.py / test_tracking.py).

def quick_overview_plots(tracks_path, out_dir, delta_t=None):
    """Original plot.py's 10-plot single-dataset overview, saved to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating velocities")
    df = pd.read_csv(tracks_path)
    df = df.dropna(subset=["track_id"]).rename(columns={"track_id": "particle"})
    df.sort_values(by=["particle", "frame"], inplace=True)

    if delta_t is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        unique_ts = pd.Series(sorted(df["timestamp"].unique()))
        delta_t = unique_ts.diff().dt.total_seconds().dropna().median()
    print(f"delta_t = {delta_t:.5f}s")

    velocities = []
    for particle_id in tqdm.tqdm(df["particle"].unique()):
        particle_data = df[df["particle"] == particle_id]
        for i in range(1, len(particle_data)):
            x1, y1, frame1, area1, esd1 = particle_data.iloc[i - 1][["x", "y", "frame", "area", "esd"]]
            x2, y2, frame2, area2, esd2 = particle_data.iloc[i][["x", "y", "frame", "area", "esd"]]

            if frame2 == frame1 + 1:
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dx = (x2 - x1) / delta_t
                dy = (y2 - y1) / delta_t
                velocity = distance / delta_t
                velocities.append({
                    "particle": particle_id, "frame1": frame1, "frame2": frame2,
                    "velocity": velocity, "dx": dx, "dy": dy,
                    "area1": area1, "area2": area2, "esd": np.mean([esd1, esd2]),
                })

    velocity_df = pd.DataFrame(velocities)
    print(velocity_df.columns)

    velocities_path = os.path.join(out_dir, "velocities.csv")
    velocity_df.to_csv(velocities_path, index=False)
    print(f"Saved velocity data to: {velocities_path}")

    if len(velocity_df) == 0:
        print("No consecutive-frame matches found — nothing to plot.")
        return

    real_matches = velocity_df[
        np.abs(velocity_df["area1"] - velocity_df["area2"]) / np.maximum(velocity_df["area1"], velocity_df["area2"]) < 0.2
    ]
    print(f"real_matches: {len(real_matches)} / {len(velocity_df)}")

    def savefig(name):
        path = os.path.join(out_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    plt.figure(figsize=(10, 6))
    plt.hist(24.0 * 3600 * 10 * real_matches["dy"] / 1_000_000, bins=30, edgecolor="black")
    plt.xlabel("Velocity in Y axis (meters/day)")
    plt.ylabel("Frequency")
    plt.title("Frequency of velocities")
    savefig("01_hist_velocity_y.png")

    positive_velocity_df = real_matches[real_matches["dy"] > 0]
    plt.figure(figsize=(10, 6))
    plt.hist(24.0 * 3600 * 10 * np.sqrt(positive_velocity_df["dy"] ** 2 + positive_velocity_df["dx"] ** 2) / 1_000_000,
             bins=30, edgecolor="black")
    plt.xlabel("Velocity (meters/day)")
    plt.ylabel("Frequency")
    plt.title("Frequency of velocities (downward only)")
    savefig("02_hist_velocity_downward.png")

    area_m2 = (10) ** 2 * real_matches["area1"] / 1e6
    velocity_m_per_day = 24.0 * 3600 * 10 * real_matches["dy"] / 1e6
    frames = real_matches["frame1"]
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(area_m2, velocity_m_per_day, c=frames, cmap="viridis", s=5, alpha=0.7)
    plt.colorbar(sc, label="Frame Number")
    plt.xlabel("Area (m^2)")
    plt.ylabel("Velocity in Y (m/day)")
    plt.title("Area vs. Y Velocity colored by Frame Number")
    plt.grid(True)
    savefig("03_area_vs_velocity_y.png")

    esd_um = np.sqrt(4 * real_matches["area1"] / np.pi) * 10
    esd_mm = esd_um / 1000
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(esd_mm, velocity_m_per_day, c=frames, cmap="magma", s=5, alpha=0.7)
    plt.colorbar(sc, label="Frame Number (increasing with time)")
    plt.xlabel("ESD (mm)")
    plt.ylabel("Velocity in Y (m/day)")
    plt.title("ESD vs. Y Velocity")
    plt.grid(True)
    savefig("04_esd_vs_velocity_y.png")

    x_area = (10) ** 2 * real_matches["area1"] / 1e6
    y_speed = 24.0 * 3600 * 10 * real_matches["dy"] / 1_000_000
    slope, intercept = np.polyfit(x_area, y_speed, 1)
    trendline = slope * x_area + intercept
    plt.figure(figsize=(10, 6))
    plt.plot(x_area, y_speed, "ro", markersize=2)
    plt.plot(x_area, trendline, "b-", label=f"Trendline: y = {slope:.2f}x + {intercept:.2f}")
    plt.ylabel("Speed in Y axis (meters/day)")
    plt.xlabel("Area (m^2)")
    plt.title("Area vs speed in Y axis")
    plt.legend()
    savefig("05_area_vs_speed_trendline.png")

    plt.figure(figsize=(10, 6))
    plt.plot((10) ** 2 * real_matches["area1"] / 1e6, (10) ** 2 * real_matches["area2"] / 1e6, "ro", markersize=2)
    plt.ylabel("Area2 (m^2)")
    plt.xlabel("Area1 (m^2)")
    plt.title("Area1 vs area2 (consecutive-frame consistency)")
    savefig("06_area1_vs_area2.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(real_matches["frame1"], 24.0 * 3600 * 10 * real_matches["dx"] / 1e6,
                c=real_matches["frame1"], cmap="plasma", s=5, alpha=0.7)
    plt.colorbar(label="Frame Number")
    plt.xlabel("Frame")
    plt.ylabel("Velocity in X (m/day)")
    plt.title("X Velocity vs Frame")
    plt.grid(True)
    savefig("07_velocity_x_vs_frame.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(real_matches["frame1"], 24.0 * 3600 * 10 * real_matches["dy"] / 1e6,
                c=real_matches["frame1"], cmap="cividis", s=5, alpha=0.7)
    plt.colorbar(label="Frame Number")
    plt.xlabel("Frame")
    plt.ylabel("Velocity in Y (m/day)")
    plt.title("Y Velocity vs Frame")
    plt.grid(True)
    savefig("08_velocity_y_vs_frame.png")

    area_min = 100
    filtered_matches = real_matches[real_matches["area1"] >= area_min]

    plt.figure(figsize=(10, 6))
    for particle_id in filtered_matches["particle"].unique():
        part_data = filtered_matches[filtered_matches["particle"] == particle_id]
        plt.plot(part_data["frame1"], 24.0 * 3600 * 10 * part_data["dx"] / 1e6,
                 marker="o", markersize=3, alpha=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Velocity in X (m/day)")
    plt.title(f"X Velocity vs Frame (lines per particle, min area {area_min})")
    plt.grid(True)
    savefig("09_velocity_x_per_particle.png")

    plt.figure(figsize=(10, 6))
    for particle_id in filtered_matches["particle"].unique():
        part_data = filtered_matches[filtered_matches["particle"] == particle_id]
        plt.plot(part_data["frame1"], 24.0 * 3600 * 10 * part_data["dy"] / 1e6,
                 marker="o", markersize=3, alpha=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Velocity in Y (m/day)")
    plt.title(f"Y Velocity vs Frame (lines per particle, min area {area_min})")
    plt.grid(True)
    savefig("10_velocity_y_per_particle.png")

    print("quick_overview_plots done.")


# ==============================================================
# SECTION 2 — full multi-dataset / time-window analysis
# ==============================================================
# Works directly off tracker output CSVs (timestamp, dx, dy, speed, area,
# track_id columns) — no manual velocity recomputation needed. Can load
# several dataset directories at once (each glob'd for "*.csv") and compare
# them side by side; also supports "brushing duration" (even-hour wiper
# exclusion window) sweeps and 10-minute time-window breakdowns.

def load_dataset(csv_root, minutes):
    csv_files = sorted(glob(os.path.join(csv_root, "*.csv")))
    all_particles = []
    all_track_counts = []
    frame_real = None

    for tracks_path in csv_files:
        try:
            df = pd.read_csv(tracks_path)

            required_cols = {"timestamp", "dx", "dy", "speed", "area", "track_id"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                print(f"Skipping {tracks_path}: missing columns {missing}")
                continue

            if df.empty:
                print(f"Skipping {tracks_path}: empty file")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Compute frame rate BEFORE filtering (diff unique frame timestamps,
            # not raw rows — a frame can have multiple detections, which would
            # otherwise make most consecutive-row diffs 0 and crash the median)
            unique_ts = pd.Series(sorted(df["timestamp"].unique()))
            time_diffs = unique_ts.diff().dt.total_seconds().dropna()
            median_diff = time_diffs.median()
            frame_rate = 1.0 / median_diff if median_diff and median_diff > 0 else np.nan

            # Filter even hours using current minutes value
            df = df[~df["timestamp"].apply(lambda ts: is_even_hour(ts, minutes))]

            # Velocidades
            df["vx_m_day"] = df["dx"]
            df["vy_m_day"] = df["dy"]
            df["speed_m_day"] = df["speed"]

            # ESD
            df["esd_um"] = np.sqrt(4 * df["area"] / np.pi) * PIXEL_SIZE_UM
            df["esd_mm"] = df["esd_um"] / 1000

        except Exception as e:
            print(f"Skipping {tracks_path}: {e}")
            continue

        counts = df["track_id"].value_counts()
        valid_particles = counts[counts > 3].index
        df_valid = df[df["track_id"].isin(valid_particles)]

        if df_valid.empty:
            print(f"Skipping {tracks_path}: no particles with more than 3 observations")
            continue
        else:
            print(f"Analizing {tracks_path}: particles with more than 3 observations")

        track_count_df = df_valid.groupby("track_id").size().reset_index(name="track_count")
        all_track_counts.append(track_count_df)

        particle_stats = (
            df_valid.groupby("track_id")
            .agg({
                "vx_m_day":  mean_without_outliers,
                "vy_m_day":  mean_without_outliers,
                "esd_mm":    mean_without_outliers,
                "timestamp": "mean",
            })
            .reset_index()
        )

        # Angle mapped to 0-360deg
        # 0deg=right, 90deg=down/sinking, 180deg=left, 270deg=up
        particle_stats["angle_deg"] = np.degrees(
            np.arctan2(particle_stats["vy_m_day"], particle_stats["vx_m_day"])
        ) % 360

        all_particles.append(particle_stats)
        frame_real = frame_rate

    if not all_particles:
        return None, None, None

    all_particles_df = pd.concat(all_particles, ignore_index=True)
    all_track_counts_df = pd.concat(all_track_counts, ignore_index=True)

    p10_threshold = all_track_counts_df["track_count"].quantile(0.1)
    all_track_counts_df["10_percent"] = all_track_counts_df["track_count"] <= p10_threshold

    return all_particles_df, all_track_counts_df, frame_real


def load_dataset_by_time_window(csv_root, window_minutes=10):
    """
    Load all CSVs and split particle observations into 10-minute windows
    measured from the start of each even hour (0, 2, 4 ...).
    Windows: 0-10, 10-20, 20-30, ..., 110-120 min after the even hour.
    Returns a dict: {window_label: particles_df}
    """
    csv_files = sorted(glob(os.path.join(csv_root, "*.csv")))
    all_rows = []

    for tracks_path in csv_files:
        try:
            df = pd.read_csv(tracks_path)
            required_cols = {"timestamp", "dx", "dy", "speed", "area", "track_id"}
            if not required_cols.issubset(df.columns):
                continue
            if df.empty:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            df["vx_m_day"]    = df["dx"]
            df["vy_m_day"]    = df["dy"]
            df["speed_m_day"] = df["speed"]
            df["esd_um"]      = np.sqrt(4 * df["area"] / np.pi) * PIXEL_SIZE_UM
            df["esd_mm"]      = df["esd_um"] / 1000

            all_rows.append(df)
        except Exception as e:
            print(f"Skipping {tracks_path}: {e}")
            continue

    if not all_rows:
        return {}

    full_df = pd.concat(all_rows, ignore_index=True)

    def minutes_since_even_hour(ts):
        last_even_hour = ts.replace(minute=0, second=0, microsecond=0)
        if ts.hour % 2 != 0:
            last_even_hour = last_even_hour - pd.Timedelta(hours=1)
        return (ts - last_even_hour).total_seconds() / 60.0

    full_df["elapsed_min"] = full_df["timestamp"].apply(minutes_since_even_hour)
    full_df = full_df[(full_df["elapsed_min"] >= 0) & (full_df["elapsed_min"] < 120)]
    full_df["window_start"] = (full_df["elapsed_min"] // window_minutes) * window_minutes

    windows = {}
    all_window_starts = list(range(0, 120, window_minutes))

    for w_start in all_window_starts:
        w_end  = w_start + window_minutes
        label  = f"{int(w_start):03d}–{int(w_end):03d} min"
        subset = full_df[full_df["window_start"] == w_start].copy()

        if subset.empty:
            continue

        counts          = subset["track_id"].value_counts()
        valid_particles = counts[counts > 3].index
        subset          = subset[subset["track_id"].isin(valid_particles)]

        if subset.empty:
            continue

        particle_stats = (
            subset.groupby("track_id")
            .agg({
                "vx_m_day":  mean_without_outliers,
                "vy_m_day":  mean_without_outliers,
                "esd_mm":    mean_without_outliers,
                "timestamp": "mean",
            })
            .reset_index()
        )
        windows[label] = particle_stats
        windows[label + "__timestamps"] = subset["timestamp"].sort_values().reset_index(drop=True)

    return windows


def plot_speed_by_time_window(windows_dict, dataset_label, bins=150, save_path=None):
    if not windows_dict:
        print("No windowed data to plot.")
        return

    all_labels = [f"{w:03d}–{w+10:03d} min" for w in range(0, 120, 10)]
    labels = [l for l in all_labels if l in windows_dict]

    base_colors = [
        "#08306b", "#08519c", "#2171b5", "#4292c6", "#41ab5d", "#238b45",
        "#006d2c", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#800026",
    ]
    color_map = {lbl: base_colors[i] for i, lbl in enumerate(all_labels)}

    shared_edges = {}
    for col in ["vy_m_day", "vx_m_day"]:
        all_vals = pd.concat([windows_dict[l][col].dropna() for l in labels])
        shared_edges[col] = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    panels = [
        ("vy_m_day", "Vertical velocity vy (m/day)"),
        ("vx_m_day", "Horizontal velocity vx (m/day)"),
    ]

    for ax, (col, xlabel) in zip(axes, panels):
        edges = shared_edges[col]
        for label in labels:
            data = windows_dict[label][col].dropna()
            if data.empty:
                continue
            counts, _ = np.histogram(data, bins=edges)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(bin_centers, counts, color=color_map[label], linewidth=1.6,
                    label=label, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=7.5, ncol=2, title="Time since\neven hour")
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Vertical velocity vy — {dataset_label} — by 10-min window")
    axes[1].set_title(f"Horizontal velocity vx — {dataset_label} — by 10-min window")
    fig.text(0.5, 0.01,
             "Colors: dark blue = just after even hour  →  dark red = 110-120 min after even hour",
             ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "speed_by_time_window.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save(save_path, filename):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches="tight")


def plot_scatter_multi(datasets_loaded, x_col, y_col, xlabel, ylabel, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    y_cols    = [y_col,          "vx_m_day"]
    ylabels   = [ylabel,         "Speed x axis (m/day)"]
    subtitles = [title + " — vy", title + " — vx"]
    for ax, yc, yl, st in zip(axes, y_cols, ylabels, subtitles):
        for ds in datasets_loaded:
            df = ds["particles_df"]
            ax.scatter(df[x_col], df[yc], c=ds["color"], edgecolors="#003366",
                       s=10, label=ds["label"], alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(yl)
        ax.set_title(st)
        ax.legend()
        ax.grid(False)
    plt.tight_layout()
    _save(save_path, "speed_vs_esd.png")
    plt.close(fig)


def plot_hist_multi(datasets_loaded, col, xlabel, title, bins=300, logx=False, save_path=None):
    fig = plt.figure(figsize=(10, 6))
    if logx:
        plt.xscale("log")
    for ds in datasets_loaded:
        data = ds["particles_df"][col].dropna()
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, color=ds["color"], label=ds["label"],
                 linewidth=2, marker="o", markersize=2)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save(save_path, f"histogram_{col}.png")
    plt.close(fig)


def plot_track_count_hist_multi(datasets_loaded, title, bins=300, save_path=None):
    fig = plt.figure(figsize=(10, 6))
    for ds in datasets_loaded:
        data = ds["track_counts_df"]["track_count"].dropna()
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, color=ds["color"], label=ds["label"],
                 linewidth=2, marker="o", markersize=2)
    plt.xlabel("Track count (number of observations per particle)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save(save_path, "track_count_distribution.png")
    plt.close(fig)


def plot_slow_particles_multi(datasets_loaded, xlabel, ylabel, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pairs = [("vy_m_day", ylabel), ("vx_m_day", "Speed x axis (m/day)")]
    for ax, (vcol, vlab) in zip(axes, pairs):
        for ds in datasets_loaded:
            ps = ds["particles_df"].groupby("track_id").mean(numeric_only=True).reset_index()
            slow_ids = ds["track_counts_df"][ds["track_counts_df"]["10_percent"] == True]["track_id"]
            slow = ps[ps["track_id"].isin(slow_ids)]
            ax.scatter(slow["esd_mm"], slow[vcol], c=ds["color"], edgecolors="#003366",
                       s=10, label=ds["label"], alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(vlab)
        ax.set_title(title + f" — {vcol}")
        ax.legend()
        ax.grid(False)
    plt.tight_layout()
    _save(save_path, "speed_vs_esd_lowest10pct.png")
    plt.close(fig)


def plot_well_tracked_particles_multi(datasets_loaded, xlabel, ylabel, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pairs = [("vy_m_day", ylabel), ("vx_m_day", "Speed x axis (m/day)")]
    for ax, (vcol, vlab) in zip(axes, pairs):
        for ds in datasets_loaded:
            ps = ds["particles_df"].groupby("track_id").mean(numeric_only=True).reset_index()
            p90 = ds["track_counts_df"]["track_count"].quantile(0.9)
            good_ids = ds["track_counts_df"][ds["track_counts_df"]["track_count"] >= p90]["track_id"]
            well = ps[ps["track_id"].isin(good_ids)]
            ax.scatter(well["esd_mm"], well[vcol], c=ds["color"], edgecolors="#003366",
                       s=10, label=ds["label"], alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(vlab)
        ax.set_title(title + f" — {vcol}")
        ax.legend()
        ax.grid(False)
    plt.tight_layout()
    _save(save_path, "speed_vs_esd_highest10pct.png")
    plt.close(fig)


def plot_scatter_vx_vy_multi(datasets_loaded, title, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    for ds in datasets_loaded:
        df = ds["particles_df"]
        plt.scatter(df["vx_m_day"], df["vy_m_day"], c=ds["color"], edgecolors="#003366",
                    s=10, label=ds["label"], alpha=0.7)
    plt.xlabel("Horizontal velocity vx (m/day)")
    plt.ylabel("Vertical velocity vy (m/day)")
    plt.title(title)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    _save(save_path, "vx_vs_vy.png")
    plt.close(fig)


def plot_angle_hist_multi(datasets_loaded, title, bins=72, save_path=None):
    fig = plt.figure(figsize=(10, 6))
    for ds in datasets_loaded:
        data = ds["particles_df"]["angle_deg"].dropna()
        counts, bin_edges = np.histogram(data, bins=bins, range=(0, 360))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, color=ds["color"], label=ds["label"],
                 linewidth=2, marker="o", markersize=2)
    plt.xlabel("Velocity angle (°)   [0°=right, 90°=down/sinking, 180°=left, 270°=up]")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(0, 361, 45))
    plt.axvline(90,  color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(270, color="gray", linewidth=0.8, linestyle=":")
    plt.text(90,  plt.ylim()[1] * 0.95, " 90° sinking", fontsize=8, color="gray", va="top")
    plt.text(270, plt.ylim()[1] * 0.95, " 270° rising", fontsize=8, color="gray", va="top")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save(save_path, "angle_distribution.png")
    plt.close(fig)


def plot_angle_polar_multi(datasets_loaded, title, bins=72, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    for ds in datasets_loaded:
        angles_rad = np.radians(ds["particles_df"]["angle_deg"].dropna())
        counts, _ = np.histogram(angles_rad, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, color=ds["color"], edgecolor="#003366",
               linewidth=0.3, alpha=0.6, label=ds["label"])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    _save(save_path, "angle_polar.png")
    plt.close(fig)


def plot_velocity_vector_vs_esd(datasets_loaded, title, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    for ds in datasets_loaded:
        df = ds["particles_df"].dropna(subset=["esd_mm", "vx_m_day", "vy_m_day"])
        speed = np.sqrt(df["vx_m_day"]**2 + df["vy_m_day"]**2)
        plt.scatter(df["esd_mm"], speed, c=ds["color"], edgecolors="#003366",
                    s=10, alpha=0.6, label=ds["label"])
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("ESD (mm)")
    plt.ylabel("Speed magnitude |v| (m/day)")
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    _save(save_path, "speed_magnitude_vs_esd.png")
    plt.close(fig)


def plot_binned_vy_vs_esd(datasets_loaded, title, n_bins=20, save_path=None):
    """Bin particles by ESD, plot median vy +/- IQR per bin, overlay Stokes settling curves."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for ds in datasets_loaded:
        df = ds["particles_df"].dropna(subset=["esd_mm", "vy_m_day"])
        df["esd_bin"] = pd.qcut(df["esd_mm"], q=n_bins, duplicates="drop")
        grouped = df.groupby("esd_bin", observed=True)["vy_m_day"]

        bin_medians = grouped.median()
        bin_q25     = grouped.quantile(0.25)
        bin_q75     = grouped.quantile(0.75)
        esd_mids    = df.groupby("esd_bin", observed=True)["esd_mm"].median()

        ax.plot(esd_mids, bin_medians, color=ds["color"], linewidth=2, marker="o",
                markersize=5, label=f"{ds['label']} — median vy")
        ax.fill_between(esd_mids, bin_q25, bin_q75, color=ds["color"], alpha=0.2,
                        label=f"{ds['label']} — IQR")

    mu        = 1.07e-3
    g         = 9.81
    esd_range = np.linspace(0.1e-3, 1.5e-3, 200)
    stokes_curves = [(1, "black", "--"), (2, "#444444", "-."), (5, "#888888", ":")]
    for rho_excess, col, ls in stokes_curves:
        v_stokes      = (rho_excess * g * esd_range**2) / (18 * mu)
        v_stokes_mday = v_stokes * SECONDS_PER_DAY
        ax.plot(esd_range * 1e3, v_stokes_mday, color=col, linewidth=1.5, linestyle=ls,
                label=f"Stokes (delta_rho={rho_excess} kg/m3)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("ESD (mm)")
    ax.set_ylabel("Vertical velocity vy — median (m/day)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(save_path, "vy_binned_vs_esd.png")
    plt.close(fig)


def plot_vy_vs_track_count(datasets_loaded, title, n_bins=20, save_path=None):
    """Median vy +/- IQR as a function of track count (binned) — helps pick a min-track-length filter."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ds in datasets_loaded:
        df = ds["particles_df"].copy()
        tc = ds["track_counts_df"][["track_id", "track_count"]]
        df = df.merge(tc, on="track_id", how="left")
        df = df.dropna(subset=["track_count", "vy_m_day", "vx_m_day"])

        for ax, vcol, ylabel in zip(
            axes, ["vy_m_day", "vx_m_day"],
            ["Vertical velocity vy (m/day)", "Horizontal velocity vx (m/day)"],
        ):
            df["tc_bin"] = pd.qcut(df["track_count"], q=n_bins, duplicates="drop")
            grouped = df.groupby("tc_bin", observed=True)

            tc_mids  = grouped["track_count"].median()
            v_median = grouped[vcol].median()
            v_q25    = grouped[vcol].quantile(0.25)
            v_q75    = grouped[vcol].quantile(0.75)
            v_std    = grouped[vcol].std()

            ax.plot(tc_mids, v_median, color=ds["color"], linewidth=2, marker="o",
                    markersize=5, label=f"{ds['label']} — median")
            ax.fill_between(tc_mids, v_q25, v_q75, color=ds["color"], alpha=0.2,
                            label=f"{ds['label']} — IQR")
            ax.plot(tc_mids, v_std, color=ds["color"], linewidth=1.5, linestyle=":",
                    marker="s", markersize=4, alpha=0.7, label=f"{ds['label']} — std")

    for ax, ylabel, vcol in zip(
        axes, ["Vertical velocity vy (m/day)", "Horizontal velocity vx (m/day)"], ["vy", "vx"],
    ):
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Track count (number of observations per particle)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {vcol}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(save_path, "vy_vs_track_count.png")
    plt.close(fig)


def plot_vy_cdf(datasets_loaded, title, save_path=None):
    """Empirical CDF of vy (and vx as a symmetry check) — read off median sinking speed directly."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ds in datasets_loaded:
        df = ds["particles_df"].dropna(subset=["vy_m_day", "vx_m_day"])
        for ax, vcol in zip(axes, ["vy_m_day", "vx_m_day"]):
            data_sorted = np.sort(df[vcol].values)
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax.plot(data_sorted, cdf, color=ds["color"], linewidth=2, label=ds["label"])

            median_val = np.median(data_sorted)
            ax.axvline(median_val, color=ds["color"], linewidth=1, linestyle="--", alpha=0.8)
            ax.text(median_val, 0.05, f" median\n {median_val:.1f} m/day",
                    color=ds["color"], fontsize=8, va="bottom")

    for ax, xlabel in zip(axes, ["Vertical velocity vy (m/day)", "Horizontal velocity vx (m/day)"]):
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(f"{title}")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(save_path, "vy_cdf.png")
    plt.close(fig)


def plot_vy_violin_by_esd_quartile(datasets_loaded, title, save_path=None):
    """Violin + box plot of vy split into log-spaced ESD bins."""
    fig, axes = plt.subplots(1, len(datasets_loaded), figsize=(8 * len(datasets_loaded), 7), sharey=False)
    if len(datasets_loaded) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets_loaded):
        df = ds["particles_df"].dropna(subset=["esd_mm", "vy_m_day"]).copy()

        esd_min  = df["esd_mm"].min()
        esd_max  = df["esd_mm"].max()
        if esd_min <= 0 or esd_min == esd_max:
            # Too little ESD spread this hour (e.g. very few particles) to form
            # distinct log-spaced bins -- skip rather than crash on duplicate edges.
            ax.set_title(f"{title} — {ds['label']} (insufficient ESD variation)")
            continue
        log_edges = np.logspace(np.log10(esd_min), np.log10(esd_max), 9)
        bin_labels = [f"{log_edges[i]:.2f}–{log_edges[i+1]:.2f} mm" for i in range(len(log_edges) - 1)]

        try:
            df["esd_logbin"] = pd.cut(df["esd_mm"], bins=log_edges, labels=bin_labels, include_lowest=True)
        except ValueError:
            ax.set_title(f"{title} — {ds['label']} (insufficient ESD variation)")
            continue

        non_empty = [lbl for lbl in bin_labels
                     if df.loc[df["esd_logbin"] == lbl, "vy_m_day"].dropna().shape[0] > 5]
        data_by_bin = [df.loc[df["esd_logbin"] == lbl, "vy_m_day"].dropna().values for lbl in non_empty]

        if not data_by_bin:
            ax.set_title(f"{title} — {ds['label']} (no data)")
            continue

        parts = ax.violinplot(data_by_bin, positions=range(len(non_empty)),
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(ds["color"])
            pc.set_alpha(0.4)
            pc.set_edgecolor("#003366")

        ax.boxplot(
            data_by_bin, positions=range(len(non_empty)), widths=0.15, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(facecolor=ds["color"], alpha=0.6),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.3, markerfacecolor=ds["color"]),
            showfliers=True,
        )

        for i, lbl in enumerate(non_empty):
            n = len(df.loc[df["esd_logbin"] == lbl, "vy_m_day"].dropna())
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5,
                    f"n={n}", ha="center", va="top", fontsize=7.5, color="gray")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(non_empty)))
        ax.set_xticklabels(non_empty, rotation=15, ha="right", fontsize=8)
        ax.set_xlabel("ESD bin (log-spaced)")
        ax.set_ylabel("Vertical velocity vy (m/day)")
        ax.set_title(f"{title} — {ds['label']}")
        ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    _save(save_path, "vy_violin_by_esd_quartile.png")
    plt.close(fig)


def write_window_timestamps_txt(windows_dict, dataset_label, save_path=None):
    all_labels = [f"{w:03d}–{w+10:03d} min" for w in range(0, 120, 10)]
    labels     = [l for l in all_labels if l in windows_dict]

    lines = [
        f"Timestamps used per 10-min time window — dataset: {dataset_label}",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]

    for label in labels:
        ts_key = label + "__timestamps"
        lines.append(f"\nWindow: {label}")
        lines.append("-" * 40)
        if ts_key in windows_dict:
            timestamps = windows_dict[ts_key]
            lines.append(f"  Total observations: {len(timestamps)}")
            lines.append(f"  First: {timestamps.iloc[0]}")
            lines.append(f"  Last:  {timestamps.iloc[-1]}")
            lines.append("  All timestamps:")
            for ts in timestamps:
                lines.append(f"    {ts}")
        else:
            lines.append("  No timestamp data available.")

    txt_content = "\n".join(lines)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fpath = os.path.join(save_path, f"window_timestamps_{dataset_label}.txt")
        with open(fpath, "w") as f:
            f.write(txt_content)
        print(f"Timestamps saved to {fpath}")
    else:
        print(txt_content)


def plot_violin_by_time_window(windows_dict, dataset_label, save_path=None):
    if not windows_dict:
        print("No windowed data to plot.")
        return

    all_labels = [f"{w:03d}–{w+10:03d} min" for w in range(0, 120, 10)]
    base_colors = [
        "#08306b", "#08519c", "#2171b5", "#4292c6", "#41ab5d", "#238b45",
        "#006d2c", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#800026",
    ]
    color_map = {lbl: base_colors[i] for i, lbl in enumerate(all_labels)}
    labels = [l for l in all_labels if l in windows_dict]

    if not labels:
        print("No windowed data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(labels) * 1.4), 8))
    panels = [
        ("vy_m_day", "Vertical velocity vy (m/day)"),
        ("vx_m_day", "Horizontal velocity vx (m/day)"),
    ]

    for ax, (col, ylabel) in zip(axes, panels):
        data_by_window, valid_labels, valid_colors = [], [], []
        for label in labels:
            data = windows_dict[label][col].dropna().values
            if len(data) > 5:
                data_by_window.append(data)
                valid_labels.append(label)
                valid_colors.append(color_map[label])

        if not data_by_window:
            ax.set_title(f"{col} — no data")
            continue

        positions = range(len(valid_labels))
        parts = ax.violinplot(data_by_window, positions=positions, showmedians=False, showextrema=False)
        for pc, col_hex in zip(parts["bodies"], valid_colors):
            pc.set_facecolor(col_hex)
            pc.set_alpha(0.55)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)

        for i, (data, col_hex) in enumerate(zip(data_by_window, valid_colors)):
            ax.boxplot(
                [data], positions=[i], widths=0.12, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                boxprops=dict(facecolor=col_hex, alpha=0.8),
                whiskerprops=dict(linewidth=1.0, color="black"),
                capprops=dict(linewidth=1.0, color="black"),
                flierprops=dict(marker=".", markersize=2, alpha=0.25, markerfacecolor=col_hex),
                showfliers=True,
            )

        for i, (label, data) in enumerate(zip(valid_labels, data_by_window)):
            ax.text(i, ax.get_ylim()[0], f"n={len(data)}", ha="center", va="top",
                    fontsize=6.5, color="gray")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Time since even hour")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split('(')[0].strip()} by 10-min window — {dataset_label}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.text(0.5, 0.01,
             "Colors: dark blue = just after even hour  →  dark red = 110-120 min after even hour",
             ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "violin_by_time_window.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_violin_30_vs_90(windows_dict, dataset_label, save_path=None):
    if not windows_dict:
        print("No windowed data to plot.")
        return

    first_30_labels = [f"{w:03d}–{w+10:03d} min" for w in range(0,  30, 10)]
    last_90_labels  = [f"{w:03d}–{w+10:03d} min" for w in range(30, 120, 10)]

    def pool_windows(label_list, col):
        parts = [windows_dict[l][col].dropna().values for l in label_list
                 if l in windows_dict and col in windows_dict[l].columns]
        return np.concatenate(parts) if parts else np.array([])

    groups = [
        ("First 30 min\n(0 – 30 min)",  first_30_labels, "#2171b5"),
        ("Last 90 min\n(30 – 120 min)", last_90_labels,  "#e31a1c"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    panels = [
        ("vy_m_day", "Vertical velocity vy (m/day)"),
        ("vx_m_day", "Horizontal velocity vx (m/day)"),
    ]

    for ax, (col, ylabel) in zip(axes, panels):
        data_by_group, valid_labels, valid_colors = [], [], []
        for group_label, label_list, color in groups:
            data = pool_windows(label_list, col)
            if len(data) > 5:
                data_by_group.append(data)
                valid_labels.append(group_label)
                valid_colors.append(color)

        if not data_by_group:
            ax.set_title(f"{col} — no data")
            continue

        positions = range(len(valid_labels))
        parts = ax.violinplot(data_by_group, positions=positions, showmedians=False, showextrema=False)
        for pc, col_hex in zip(parts["bodies"], valid_colors):
            pc.set_facecolor(col_hex)
            pc.set_alpha(0.45)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.7)

        for i, (data, col_hex) in enumerate(zip(data_by_group, valid_colors)):
            ax.boxplot(
                [data], positions=[i], widths=0.14, patch_artist=True,
                medianprops=dict(color="black", linewidth=2.5),
                boxprops=dict(facecolor=col_hex, alpha=0.75),
                whiskerprops=dict(linewidth=1.0, color="black"),
                capprops=dict(linewidth=1.0, color="black"),
                flierprops=dict(marker=".", markersize=2, alpha=0.25, markerfacecolor=col_hex),
                showfliers=True,
            )

        for i, (data, col_hex) in enumerate(zip(data_by_group, valid_colors)):
            median = np.median(data)
            q25    = np.percentile(data, 25)
            q75    = np.percentile(data, 75)
            ylim   = ax.get_ylim()
            y_ann  = ylim[0] + (ylim[1] - ylim[0]) * 0.02
            ax.text(i, y_ann, f"n={len(data)}\nmedian={median:.1f}\nIQR=[{q25:.1f}, {q75:.1f}]",
                    ha="center", va="bottom", fontsize=8, color="gray")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(valid_labels, fontsize=10)
        ax.set_xlabel("Time window group")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split('(')[0].strip()} — first 30 vs last 90 min — {dataset_label}")
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"First 30 min vs last 90 min after even hour — {dataset_label}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "violin_30_vs_90min.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speed_hist_all_minutes(all_minutes_data, bins=300, save_path=None):
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    panels = [
        ("vy_m_day", "Vertical velocity vy (m/day)",   "Frequency vs vy — all brushing durations"),
        ("vx_m_day", "Horizontal velocity vx (m/day)", "Frequency vs vx — all brushing durations"),
    ]

    for ax, (col, xlabel, subtitle) in zip(axes, panels):
        for i, (minutes, loaded) in enumerate(all_minutes_data.items()):
            color = colors[i % len(colors)]
            for ds in loaded:
                data = ds["particles_df"][col].dropna()
                counts, bin_edges = np.histogram(data, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.plot(bin_centers, counts, color=color, linewidth=1.8,
                        label=f"{ds['label']} — {minutes} min", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(subtitle)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.suptitle("Speed distributions — comparison across brushing durations", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "speed_hist_all_minutes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_full_analysis(datasets, minutes_list, out_root):
    """Run the full brushing-duration sweep + time-window analysis for one or more datasets."""
    all_minutes_data = {}

    for minutes in minutes_list:
        print(f"\n{'='*50}")
        print(f"Plotting for brushing until {minutes} min")
        print(f"{'='*50}")

        save_dir = os.path.join(out_root, f"{minutes}min")
        os.makedirs(save_dir, exist_ok=True)

        loaded = []
        for ds in datasets:
            print(f"Loading {ds['label']}...")
            particles_df, track_counts_df, frame_real = load_dataset(ds["path"], minutes)
            if particles_df is not None:
                loaded.append({
                    "particles_df": particles_df, "track_counts_df": track_counts_df,
                    "frame_real": frame_real, "color": ds["color"], "label": ds["label"],
                })
            else:
                print(f"No valid data for {ds['label']}")

        if loaded:
            plot_scatter_multi(loaded, x_col="esd_mm", y_col="vy_m_day",
                                xlabel="ESD (mm)", ylabel="Speed y axis (m/day)",
                                title=f"Speed vs ESD — brushing until {minutes} min", save_path=save_dir)
            plot_scatter_vx_vy_multi(loaded, title=f"Horizontal vs Vertical velocity — brushing until {minutes} min", save_path=save_dir)
            plot_velocity_vector_vs_esd(loaded, title=f"Speed magnitude vs ESD — brushing until {minutes} min", save_path=save_dir)
            plot_angle_hist_multi(loaded, title=f"Velocity angle distribution — brushing until {minutes} min", save_path=save_dir)
            plot_angle_polar_multi(loaded, title=f"Velocity direction (polar) — brushing until {minutes} min", save_path=save_dir)
            plot_track_count_hist_multi(loaded, title=f"Track Count Distribution — brushing until {minutes} min", save_path=save_dir)
            plot_slow_particles_multi(loaded, xlabel="ESD (mm)", ylabel="Speed y axis (m/day)",
                                       title=f"Speed vs ESD for 10% lowest track count — brushing until {minutes} min", save_path=save_dir)
            plot_well_tracked_particles_multi(loaded, xlabel="ESD (mm)", ylabel="Vertical velocity vy (m/day)",
                                               title=f"Speed vs ESD for 10% highest track count — brushing until {minutes} min", save_path=save_dir)
            plot_hist_multi(loaded, col="esd_mm", xlabel="ESD (mm)",
                             title=f"ESD Distribution — brushing until {minutes} min", save_path=save_dir)
            plot_hist_multi(loaded, col="vy_m_day", xlabel="Vertical velocity vy (m/day)",
                             title=f"Vertical Velocity Distribution — brushing until {minutes} min", save_path=save_dir)
            plot_hist_multi(loaded, col="vx_m_day", xlabel="Horizontal velocity vx (m/day)",
                             title=f"Horizontal Velocity Distribution — brushing until {minutes} min", save_path=save_dir)
            plot_binned_vy_vs_esd(loaded, title=f"Binned median vy vs ESD — brushing until {minutes} min", n_bins=20, save_path=save_dir)
            plot_vy_vs_track_count(loaded, title=f"vy vs track count — brushing until {minutes} min", n_bins=20, save_path=save_dir)
            plot_vy_cdf(loaded, title=f"CDF of velocity — brushing until {minutes} min", save_path=save_dir)
            plot_vy_violin_by_esd_quartile(loaded, title=f"vy by ESD quartile — brushing until {minutes} min", save_path=save_dir)

            all_minutes_data[minutes] = loaded
        else:
            print(f"No valid data for brushing until {minutes} min")

    if all_minutes_data:
        print("\nPlotting cross-minutes speed frequency comparison...")
        plot_speed_hist_all_minutes(all_minutes_data, bins=300, save_path=os.path.join(out_root, "comparison"))

    print("\nPlotting speed distributions by 10-min time window...")
    for ds in datasets:
        windows = load_dataset_by_time_window(ds["path"], window_minutes=10)
        if windows:
            window_save_dir = os.path.join(out_root, "time_windows")
            plot_speed_by_time_window(windows, dataset_label=ds["label"], bins=150, save_path=window_save_dir)
            plot_violin_by_time_window(windows, dataset_label=ds["label"], save_path=window_save_dir)
            plot_violin_30_vs_90(windows, dataset_label=ds["label"], save_path=window_save_dir)
            write_window_timestamps_txt(windows, dataset_label=ds["label"], save_path=window_save_dir)
        else:
            print(f"No windowed data for {ds['label']}")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    QUICK_TRACKS_CSV = "/mnt/Tempbox/fernanda/Sinker/tracking_imporoved_out/tracked.csv"
    QUICK_OUT_DIR    = "/mnt/Tempbox/fernanda/Sinker/plots"

    FULL_DATASETS = [
        {"path": "/mnt/Tempbox/fernanda/Sinker/tracking_imporoved_out/", "color": "tomato", "label": "first200_18"},
    ]
    FULL_MINUTES_LIST = [20]
    FULL_OUT_ROOT = "/mnt/Tempbox/fernanda/Sinker/full_analysis"

    quick_overview_plots(QUICK_TRACKS_CSV, QUICK_OUT_DIR)
    run_full_analysis(FULL_DATASETS, FULL_MINUTES_LIST, FULL_OUT_ROOT)
