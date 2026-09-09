import argparse
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "detection"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tracking"))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis"))

from old_detector import process_shadowgraph_folder
from tracking_improved import (
    track_particles_from_dataframe,
    smooth_track_velocities,
    visualise_frames_cv2,
    make_video,
)
from plot import (
    quick_overview_plots,
    plot_vy_violin_by_esd_quartile,
    mean_without_outliers,
    is_even_hour,
    PIXEL_SIZE_UM,
)

# Brushing-duration exclusion window for the ESD violin plot: data collected in
# the first VIOLIN_BRUSHING_MINUTES after each even hour (wiper artifact) is
# dropped. Matches the value plot.py's own __main__ block uses (FULL_MINUTES_LIST = [20]).
VIOLIN_BRUSHING_MINUTES = 20

# Minimum ESD (equivalent spherical diameter, mm) kept for plotting.
# PIXEL_SIZE_UM (imported from plot.py) is 10 um/pixel, which is specifically
# the Shadowgraph camera's scale (HM=2.7, LM=20 -- different pipeline if we
# ever process those). Detection/tracking themselves are NOT filtered -- the
# combined CSV still has every detected particle; this only limits what goes
# into the plots.
DEFAULT_MIN_ESD_MM = 1.0


def build_particles_df(df, minutes):
    """
    Per-track (vx, vy, esd_mm) stats for a single already-filtered tracking
    dataframe, in memory -- mirrors plot.py's load_dataset() aggregation logic
    but works directly on a dataframe instead of glob'ing a directory of CSVs.
    (load_dataset would be unsafe here: with everything flattened into one
    output_dir, its *.csv glob would also match the combined detections+
    tracking CSV, which now satisfies its required-columns check too --
    double-counting every particle.)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["vx_m_day"] = df["dx"]
    df["vy_m_day"] = df["dy"]
    df["esd_um"] = np.sqrt(4 * df["area"] / np.pi) * PIXEL_SIZE_UM
    df["esd_mm"] = df["esd_um"] / 1000

    df = df[~df["timestamp"].apply(lambda ts: is_even_hour(ts, minutes))]

    counts = df["track_id"].value_counts()
    valid_particles = counts[counts > 3].index
    df_valid = df[df["track_id"].isin(valid_particles)]
    if df_valid.empty:
        return None

    return (
        df_valid.groupby("track_id")
        .agg({
            "vx_m_day": mean_without_outliers,
            "vy_m_day": mean_without_outliers,
            "esd_mm": mean_without_outliers,
            "timestamp": "mean",
        })
        .reset_index()
    )

# Tracking parameters tuned against the particles_v3 synthetic ground truth —
# see tracking_improved.py's own __main__ block for the full rationale.
ALPHA = 1.0
BETA = 1500.0
GAMMA = 200.0
MAX_DISTANCE = 2200
AREA_THRESHOLD = 0
MAX_MISSING = 2
MAX_AREA_RATIO = 1.3
METRES_PER_PIXEL = 10e-6

DEFAULT_OUTPUT_ROOT = "/mnt/Durkin_Data/SINKER_processed"
SHADOWGRAPH_DIR_RE = re.compile(r"^Shadowgraph_\d+$")


def derive_output_dir(images_dir, output_root=DEFAULT_OUTPUT_ROOT):
    """
    Re-root images_dir under output_root, keeping only the Shadowgraph_<id>/.../
    tail of the path (everything from the Shadowgraph_<id> folder onward).

    e.g. .../SINKER/MBARI2615/MAC/Volumes/SINKER/MARS/Shadowgraph_40297765/2026/05/01/22
      -> <output_root>/Shadowgraph_40297765/2026/05/01/22
    """
    parts = os.path.normpath(images_dir).split(os.sep)
    for i, part in enumerate(parts):
        if SHADOWGRAPH_DIR_RE.match(part):
            return os.path.join(output_root, *parts[i:])
    raise ValueError(
        f"Could not find a Shadowgraph_<id> folder in images_dir path: {images_dir!r}"
    )


def run_pipeline(images_dir, output_dir, batch_size=40, window_size=20,
                  n_cpu_workers=None, make_visualizations=False, video_fps=5,
                  make_plots=True, min_esd_mm=DEFAULT_MIN_ESD_MM, save_rois=False):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[1/2] Detection (old_detector): {images_dir} -> {output_dir}")
    process_shadowgraph_folder(
        shadowgraph_path=images_dir,
        save_root=output_dir,
        batch_size=batch_size,
        window_size=window_size,
        n_cpu_workers=n_cpu_workers,
        save_rois=save_rois,
    )

    folder_name = os.path.basename(os.path.normpath(images_dir))

    # old_detector names its output after the first frame's timestamp (not the
    # folder name), so just grab whatever CSV it wrote to output_dir.
    candidates = glob.glob(os.path.join(output_dir, "*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No detections CSV found in {output_dir}")
    detections_csv = max(candidates, key=os.path.getmtime)
    logger.info(f"Detections CSV: {detections_csv}")

    logger.info("[2/2] Tracking (tracking_improved)")
    df = pd.read_csv(detections_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    frame_times = df[["image_path", "timestamp"]].drop_duplicates().sort_values("timestamp")
    dt_median = frame_times["timestamp"].diff().dropna().median().total_seconds()
    df["frame_rate"] = 1.0 / dt_median

    img_w = int(df["image_width"].iloc[0]) if "image_width" in df.columns else 4600
    img_h = int(df["image_height"].iloc[0]) if "image_height" in df.columns else 4000
    phys_max_dist = min(MAX_DISTANCE * dt_median, 0.9 * np.sqrt(img_w ** 2 + img_h ** 2))

    df_tracked, _ = track_particles_from_dataframe(
        df,
        max_distance=phys_max_dist,
        max_missing=MAX_MISSING,
        area_threshold=AREA_THRESHOLD,
        dt=dt_median,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        img_w=img_w,
        img_h=img_h,
        first_track_id=0,
        max_area_ratio=MAX_AREA_RATIO,
    )
    df_tracked = smooth_track_velocities(df_tracked)

    # df_tracked is the same rows as detections_csv plus track_id/dx/dy/speed/
    # frame_rate -- a strict superset, no information dropped -- so overwrite
    # the same timestamp-named file instead of keeping a separate tracked.csv.
    df_tracked.to_csv(detections_csv, index=False)
    n_tracks = df_tracked["track_id"].nunique(dropna=True)
    logger.info(f"Combined detections+tracking CSV -> {detections_csv}  ({n_tracks} tracks)")

    if make_visualizations:
        frames_dir = os.path.join(output_dir, "frames")
        visualise_frames_cv2(
            df_tracked, frames_dir, max_frames=500, min_area=0,
            metres_per_pixel=METRES_PER_PIXEL, arrow_scale=0.2, max_arrow_px=150.0,
        )
        video_path = os.path.join(output_dir, "tracking_video.mp4")
        make_video(frames_dir, video_path, fps=video_fps)
        logger.info(f"Video -> {video_path}")

    if make_plots:
        n_before = len(df_tracked)
        esd_mm = np.sqrt(4 * df_tracked["area"] / np.pi) * PIXEL_SIZE_UM / 1000
        plot_df = df_tracked[esd_mm >= min_esd_mm]
        logger.info(
            f"[Plots] Filtering to ESD >= {min_esd_mm}mm (Shadowgraph, {PIXEL_SIZE_UM}um/px): "
            f"{len(plot_df)}/{n_before} rows kept"
        )

        logger.info(f"[Plots] Generating overview plots -> {output_dir}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            # quick_overview_plots needs a CSV path; the filtered subset is only
            # ever needed transiently here, so it doesn't get left behind in
            # output_dir (which would otherwise re-trigger the double-count
            # problem build_particles_df's docstring explains).
            plot_input_csv = os.path.join(tmp_dir, "tracked_filtered.csv")
            plot_df.to_csv(plot_input_csv, index=False)
            quick_overview_plots(plot_input_csv, output_dir)

        # quick_overview_plots also writes its own intermediate velocities.csv
        # into output_dir -- not wanted in the final output, only the plots.
        velocities_csv = os.path.join(output_dir, "velocities.csv")
        if os.path.exists(velocities_csv):
            os.remove(velocities_csv)

        particles_df = build_particles_df(plot_df, minutes=VIOLIN_BRUSHING_MINUTES)
        if particles_df is not None:
            logger.info(f"[Plots] Generating ESD violin plot -> {output_dir}")
            plot_vy_violin_by_esd_quartile(
                [{"particles_df": particles_df, "color": "tomato", "label": folder_name}],
                title=f"vy by ESD quartile — {folder_name}",
                save_path=output_dir,
            )
        else:
            logger.warning("[Plots] Not enough tracked particles (>3 obs) for ESD violin plot — skipping")

    return detections_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect (old_detector) then track (tracking_improved) particles in one Shadowgraph folder."
    )
    parser.add_argument("images_dir", help="Folder of raw *.jpeg frames")
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help=(
            "Where detections CSV, ROIs, tracked CSV, and video go. If omitted, "
            f"derived from images_dir as {DEFAULT_OUTPUT_ROOT}/Shadowgraph_<id>/..."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--cpu-workers", type=int, default=None)
    parser.add_argument("--viz", action="store_true", help="Also render frames + tracking video (off by default)")
    parser.add_argument("--no-plots", action="store_true", help="Skip overview plots")
    parser.add_argument(
        "--save-rois", action="store_true",
        help="Also save per-particle ROI crops + per-frame contour overlay PNGs during detection (off by default)",
    )
    parser.add_argument(
        "--min-esd-mm", type=float, default=DEFAULT_MIN_ESD_MM,
        help="Minimum ESD (mm) kept for plotting -- detection/tracking still run on everything, "
             "this only limits what goes into the plots",
    )
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or derive_output_dir(args.images_dir)
    logger.info(f"Output dir: {output_dir}")

    run_pipeline(
        images_dir=args.images_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        n_cpu_workers=args.cpu_workers,
        make_visualizations=args.viz,
        video_fps=args.fps,
        make_plots=not args.no_plots,
        min_esd_mm=args.min_esd_mm,
        save_rois=args.save_rois,
    )
