import cv2
import sys
import os
import uuid
import warnings
import time
import glob
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger
import cupy as cp


# ==============================================================
# METADATA
# ==============================================================

def is_even_hour(ts):
    return ts.hour % 2 == 0 and ts.minute < 10


def extract_metadata(image_path):
    filename = os.path.basename(image_path)
    pattern = r"^([^_]+)_(\d{8}T\d{6}\.\d+).*?_(\w+)_"
    m = re.search(pattern, filename)

    if not m:
        return None, None, None, None, None

    instrument = m.group(1)
    timestamp_str = m.group(2)
    camera = m.group(3)

    clean = timestamp_str.replace("T", "")
    dt = datetime.strptime(clean, "%Y%m%d%H%M%S.%f")
    ts = pd.Timestamp(dt)
    wipe = is_even_hour(ts)

    return instrument, timestamp_str, camera, ts, wipe


# ==============================================================
# IMAGE LOADING  (parallelized)
# ==============================================================

def _load_image(p):
    """Load a single image as a numpy array."""
    return np.array(Image.open(p))


def load_images_parallel(paths, n_workers=None):
    """
    Load images concurrently with threads.
    Threads are ideal here: disk I/O releases the GIL, there is zero
    pickling/process-spawn overhead, and numpy arrays stay in shared memory.
    """
    n_workers = n_workers or min(32, len(paths))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_load_image, p) for p in paths]
        return [f.result() for f in futures]


# ==============================================================
# BACKGROUND SUBTRACTION  (GPU — stays in main process)
# ==============================================================

def compute_rolling_background(all_paths, batch_start, window_size, n_workers=None):
    """
    Compute a max-pixel background from a symmetric window of frames
    centered on the current batch position.

    The window is clamped to [0, len(all_paths) - 1] so the first and
    last batches naturally use an asymmetric (one-sided) window instead
    of failing or wrapping.

    Parameters
    ----------
    all_paths   : list[str]  — full sorted list of frame paths for this folder
    batch_start : int        — index of the first frame in the current batch
    window_size : int        — total number of frames to include in the window
    n_workers   : int | None — threads for parallel image loading

    Returns
    -------
    numpy ndarray  — background image (CPU), same dtype/shape as a single frame
    """
    half = window_size // 2
    lo = max(0, batch_start - half)
    hi = min(len(all_paths), batch_start + half + (window_size % 2))
    window_paths = all_paths[lo:hi]

    window_images = load_images_parallel(window_paths, n_workers=n_workers)

    pool = cp.get_default_memory_pool()
    try:
        stack = cp.stack([cp.asarray(img) for img in window_images], axis=0)
        background_gpu = cp.median(stack, axis=0)
        background_cpu = cp.asnumpy(background_gpu)
        del stack, background_gpu
    finally:
        pool.free_all_blocks()

    return background_cpu  # plain numpy array


def _subtract_one_frame_cpu(img_cpu, static_bg_cpu):
    """Pure-numpy fallback when GPU has no memory at all."""
    return np.abs(
        img_cpu.astype(np.int16) - static_bg_cpu.astype(np.int16)
    ).clip(0, 255).astype(np.uint8)


def background_subtraction_batch(images, static_bg_cpu, gpu_chunk_size=4):
    """
    OOM-safe GPU background subtraction.

    static_bg_cpu is a plain numpy array (lives on CPU between calls).
    For each chunk we push background + chunk to GPU, compute diff,
    pull result back, then immediately free all GPU memory.

    gpu_chunk_size=4 is conservative; raise it if you have more VRAM.
    Falls back to frame-by-frame GPU, then pure CPU numpy if still OOM.
    """
    pool = cp.get_default_memory_pool()
    results = []

    for start in range(0, len(images), gpu_chunk_size):
        chunk = images[start: start + gpu_chunk_size]
        try:
            bg_gpu    = cp.asarray(static_bg_cpu)
            batch_gpu = cp.stack([cp.asarray(img) for img in chunk], axis=0)
            diff      = cp.abs(
                batch_gpu.astype(cp.int16) - bg_gpu.astype(cp.int16)
            ).clip(0, 255).astype(cp.uint8)
            diff_cpu  = cp.asnumpy(diff)
            results.extend([diff_cpu[i] for i in range(diff_cpu.shape[0])])
            del bg_gpu, batch_gpu, diff, diff_cpu
            pool.free_all_blocks()

        except cp.cuda.memory.OutOfMemoryError:
            logger.warning(f"GPU OOM on chunk of {len(chunk)}, retrying frame-by-frame on GPU")
            pool.free_all_blocks()
            for img_cpu in chunk:
                try:
                    bg_gpu  = cp.asarray(static_bg_cpu)
                    img_gpu = cp.asarray(img_cpu)
                    diff    = cp.abs(
                        img_gpu.astype(cp.int16) - bg_gpu.astype(cp.int16)
                    ).clip(0, 255).astype(cp.uint8)
                    results.append(cp.asnumpy(diff))
                    del bg_gpu, img_gpu, diff
                    pool.free_all_blocks()
                except cp.cuda.memory.OutOfMemoryError:
                    logger.warning("GPU OOM on single frame, falling back to CPU numpy")
                    pool.free_all_blocks()
                    results.append(_subtract_one_frame_cpu(img_cpu, static_bg_cpu))

    return results


# ==============================================================
# ROI DETECTION  (CPU — safe to run in worker processes)
# ==============================================================

def find_bright_regions_df(original_image, binary_mask, save_dir, save=True, image_name=None):
    if len(original_image.shape) == 3:
        image_with_contours = original_image.copy()
    else:
        image_with_contours = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0 or area <= 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        xx, yy = x + w, y + h
        center_x, center_y = (x + xx) / 2, (y + yy) / 2
        larger = max(w, h)

        h_img, w_img = original_image.shape[:2]
        x1 = max(0, int(center_x - larger / 2))
        y1 = max(0, int(center_y - larger / 2))
        x2 = min(w_img, int(center_x + larger / 2))
        y2 = min(h_img, int(center_y + larger / 2))
        roi = original_image[y1:y2, x1:x2]

        perimeter = cv2.arcLength(cnt, True)
        esd = 2 * np.sqrt(area / np.pi)
        roi_id = str(uuid.uuid4())
        roi_filename = f"{roi_id}_roi_{int(area)}.png"

        roi_dir = (
            os.path.join(save_dir, "ROIs", image_name)
            if image_name
            else os.path.join(save_dir, "ROIs")
        )
        roi_path = os.path.join(roi_dir, roi_filename) if save else np.nan

        if save and roi.size > 0:
            os.makedirs(roi_dir, exist_ok=True)
            success = cv2.imwrite(roi_path, roi)
            if not success:
                logger.warning(f"Failed to save ROI at {roi_path}")

        data.append({
            'detection_id': roi_id,
            'area': area,
            'x': x, 'y': y, 'w': w, 'h': h,
            'xx': xx, 'yy': yy,
            'perimeter': perimeter,
            'esd': esd,
            'roi_path': roi_path,
        })

    if save:
        contour_dir = os.path.join(save_dir, "contours")
        os.makedirs(contour_dir, exist_ok=True)
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        drawn = cv2.drawContours(image_with_contours, large_contours, -1, (0, 0, 255), 2)
        contour_filename = (
            f"{image_name}_contours.png" if image_name else f"{uuid.uuid4()}_contours.png"
        )
        cv2.imwrite(os.path.join(contour_dir, contour_filename), drawn)

    return pd.DataFrame(data)

def locate(fg_mask, original_image, image_path, save=True, save_dir=""):
    
    canny = cv2.Canny(fg_mask, 50, 70, apertureSize=3, L2gradient=True)
    kernel = np.ones((10, 10), np.uint8)
    canny = cv2.dilate(canny, kernel, 2)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(canny)
    cv2.drawContours(mask, contours, -1, 255, -1)
    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.erode(mask, kernel, 2)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return find_bright_regions_df(original_image, mask, save_dir, save=save, image_name=image_name)


# ==============================================================
# FRAME WORKER  (one frame — called by ThreadPoolExecutor)
# ==============================================================

_EMPTY_ROW = {
    'detection_id': None, 'area': None,
    'x': None, 'y': None, 'w': None, 'h': None,
    'xx': None, 'yy': None,
    'perimeter': None, 'esd': None, 'roi_path': None,
}


def _locate_one(fg_mask, original_image, image_path, save, save_dir, frame_index):
    """
    Process a single frame: run locate() and attach metadata.
    OpenCV releases the GIL, so this runs truly parallel inside a ThreadPoolExecutor
    with zero pickling or process-spawn overhead.
    """
    instrument, _, camera, dt, wipe = extract_metadata(image_path)
    height, width = original_image.shape[:2]

    if wipe:
        features = pd.DataFrame([_EMPTY_ROW.copy()])
    else:
        features = locate(fg_mask, original_image, image_path, save=save, save_dir=save_dir)
        if len(features) == 0:
            features = pd.DataFrame([_EMPTY_ROW.copy()])

    features["frame"]        = frame_index
    features["image_path"]   = image_path
    features["image_width"]  = width
    features["image_height"] = height
    features["timestamp"]    = dt
    features["camera"]       = camera
    features["instrument"]   = instrument
    features["wipe"]         = wipe
    features["class"]        = "unknown"
    features["label"]        = "unknown"
    features["model"]        = "unknown"
    features["track_id"]     = None
    features["dx"]           = np.nan
    features["dy"]           = np.nan
    features["speed"]        = np.nan

    return features


# ==============================================================
# BATCH  (threaded over frames — no pickling, OpenCV releases GIL)
# ==============================================================

def batch(
    original_frames,
    fg_masks,
    save_dir,
    image_paths,
    index,
    save=True,
    n_workers=None,
):
    """
    Run locate() on every frame concurrently using threads.

    Why threads and not processes here?
    - OpenCV (Canny, dilate, findContours, erode) releases the GIL → real parallelism.
    - No pickling of large numpy arrays between processes → no serialization tax.
    - Workers are already alive (thread pool) → no spawn/fork latency per batch.
    """
    n_workers = n_workers or os.cpu_count()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _locate_one,
                fg_masks[i], original_frames[i], image_paths[i],
                save, save_dir, index + i,
            )
            for i in range(len(original_frames))
        ]
        frame_dfs = [f.result() for f in futures]

    if frame_dfs:
        result = pd.concat(frame_dfs, ignore_index=True)
        result["frame"] = result["frame"].astype(int)
        return result

    warnings.warn("No features found in any frame.")
    return pd.DataFrame(columns=[
        'detection_id', 'class', 'area', 'x', 'y', 'w', 'h',
        'xx', 'yy', 'perimeter', 'esd', 'roi_path',
        'frame', 'image_path', 'image_width', 'image_height',
        'timestamp', 'camera', 'instrument', 'wipe',
        'label', 'model', 'track_id', 'dx', 'dy', 'speed',
    ])


# ==============================================================
# FOLDER PROCESSING
# ==============================================================

def process_shadowgraph_folder(
    shadowgraph_path,
    save_root,
    batch_size=40,
    window_size=20,
    n_cpu_workers=None,
    save_rois=False,
):
    """
    Process one folder of shadowgraph images.

    Parallelism layout
    ------------------
    - Image loading       → parallel CPU  (threads)
    - Background subtract → serial  GPU   (main process, CuPy)
    - locate() per frame  → parallel CPU  (threads inside batch())

    Background computation
    ----------------------
    For each batch the background is recomputed from a rolling window of
    `window_size` frames centred on `batch_start`.  This replaces the old
    static background that was computed once from the first window_size
    frames.  The window is clamped to the folder's frame range, so the
    first and last batches use an asymmetric (one-sided) window rather
    than failing.
    """
    n_cpu_workers = n_cpu_workers or os.cpu_count()
    logger.info(f"Processing: {shadowgraph_path}  (cpu_workers={n_cpu_workers})")
    start_total = time.time()

    image_paths = sorted(glob.glob(os.path.join(shadowgraph_path, "*.jpeg")))
    if not image_paths:
        logger.error("No JPEG images found.")
        return

    folder_name = os.path.basename(os.path.dirname(image_paths[0]))
    logger.info(f"Folder: {folder_name}")

    # Name outputs after the first frame's full timestamp (not just the hour
    # folder name, e.g. "00") so they stay unique/identifiable outside the
    # nested year/month/day/hour directory structure.
    _, _, _, first_ts, _ = extract_metadata(image_paths[0])
    if first_ts is not None:
        run_id = first_ts.strftime("%Y%m%dT%H%M%S")
    else:
        run_id = folder_name
        logger.warning(
            "Could not parse a timestamp from the first image's filename; "
            "falling back to the folder name for output filenames."
        )

    if len(image_paths) < window_size:
        logger.error(f"Need at least {window_size} images for background window.")
        return

    output_path     = os.path.join(save_root, f"{run_id}.csv")
    output_time_txt = os.path.join(save_root, f"{run_id}.txt")

    if os.path.exists(output_path) and os.path.exists(output_time_txt):
        logger.info(f"Skipping {run_id}: outputs already exist.")
        return
    if os.path.exists(output_path) and not os.path.exists(output_time_txt):
        logger.warning(f"Incomplete outputs for {run_id}, reprocessing.")
        os.remove(output_path)

    # ── Frame rate ──────────────────────────────────────────────
    timestamps = []
    for p in image_paths:
        _, _, _, ts, _ = extract_metadata(p)
        if ts is not None:
            timestamps.append(ts)

    if len(timestamps) > 1:
        timestamps_sorted = sorted(timestamps)
        diffs = [
            (timestamps_sorted[i + 1] - timestamps_sorted[i]).total_seconds()
            for i in range(len(timestamps_sorted) - 1)
        ]
        median_diff = np.median(diffs)
        frame_rate = 1.0 / median_diff if median_diff > 0 else None
        logger.info(f"Frame rate: {frame_rate:.3f} Hz")
    else:
        frame_rate = None
        logger.warning("Not enough timestamps to compute frame rate.")

    # ── Main loop ───────────────────────────────────────────────
    all_batch_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc=folder_name):
        batch_paths = image_paths[i: i + batch_size]

        # 1. Load batch images — parallel CPU
        t0 = time.time()
        images = load_images_parallel(batch_paths, n_workers=n_cpu_workers)
        logger.info(f"Image loading:          {time.time() - t0:.2f}s  ({len(batch_paths)} frames)")

        # 2. Rolling background — load window + GPU max-pixel
        #    The window is centred on batch_start (i) and clamped to [0, N).
        t0 = time.time()
        rolling_bg = compute_rolling_background(
            all_paths=image_paths,
            batch_start=i,
            window_size=window_size,
            n_workers=n_cpu_workers,
        )
        logger.info(f"Rolling background:     {time.time() - t0:.2f}s  "
                    f"(window centred on frame {i}, size {window_size})")

        # 3. Background subtraction — GPU (main process only)
        t0 = time.time()
        fg_masks = background_subtraction_batch(images, rolling_bg)
        logger.info(f"Background subtraction: {time.time() - t0:.2f}s")

        # 4. Locate features — parallel CPU
        t0 = time.time()
        features = batch(
            images,
            fg_masks,
            save_dir=save_root,
            image_paths=batch_paths,
            index=i,
            save=save_rois,
            n_workers=n_cpu_workers,
        )
        logger.info(f"Feature extraction:     {time.time() - t0:.2f}s")

        features["frame_rate"] = frame_rate
        all_batch_features.append(features)

    # ── Write outputs ────────────────────────────────────────────
    os.makedirs(save_root, exist_ok=True)
    full_df = pd.concat(all_batch_features, ignore_index=True)
    full_df.to_csv(output_path, index=False)
    logger.info(f"CSV written: {output_path}  ({len(full_df)} rows)")

    total_time = time.time() - start_total
    logger.info(f"Total time: {total_time:.1f}s  ({total_time / len(image_paths):.4f}s/image)")

    with open(output_time_txt, "w") as f:
        f.write(f"Folder: {folder_name}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Images processed: {len(image_paths)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"CPU workers: {n_cpu_workers}\n")
        f.write(
            f"Frame rate (Hz): {frame_rate:.3f}\n" if frame_rate else "Frame rate (Hz): unknown\n"
        )
        f.write(f"Total time (seconds): {total_time:.2f}\n")
        f.write(f"Total time (minutes): {total_time / 60:.2f}\n")
        f.write(f"Time per image (seconds): {total_time / len(image_paths):.4f}\n")


# ==============================================================
# FOLDER WORKER  (used by the top-level ProcessPoolExecutor)
# ==============================================================

def _process_one_folder(args):
    folder, save_root, batch_size, window_size, n_cpu_workers = args
    try:
        process_shadowgraph_folder(
            shadowgraph_path=folder,
            save_root=save_root,
            batch_size=batch_size,
            window_size=window_size,
            n_cpu_workers=n_cpu_workers,
        )
        return folder, None
    except Exception as e:
        logger.exception(f"Error in folder {folder}: {e}")
        return folder, e


# ==============================================================
# DATE FILTER
# ==============================================================

def _parse_folder_date(folder_path):
    name = os.path.basename(folder_path)
    m = re.search(r"(\d{8})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
    return None


# ==============================================================
# MAIN ENTRY
# ==============================================================

def process_all_shadowgraph_folders(
    root_shadowgraph_path,
    save_root,
    batch_size=50,
    window_size=20,
    n_folder_workers=1,
    n_cpu_workers=None,
    start_date=None,
    end_date=None,
):
    """
    Process all shadowgraph folders.

    Parameters
    ----------
    n_folder_workers : int
        Number of folders to process in parallel.
        Default 1 (folders processed one at a time, frames parallelized within).
        Set > 1 only if you have many folders AND enough CPU cores to share.
        WARNING: each folder already saturates all CPUs via n_cpu_workers,
        so n_folder_workers > 1 is only useful when n_cpu_workers is limited.
    n_cpu_workers : int | None
        CPU workers used inside each folder for image loading and locate().
        None → os.cpu_count() (use all cores).
    window_size : int
        Number of frames in the rolling background window centred on each
        batch.  Larger values produce a more stable background estimate at
        the cost of loading more frames per batch.
    """
    n_cpu_workers = n_cpu_workers or os.cpu_count()

    start = datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
    end   = datetime.strptime(end_date,   "%Y%m%d").date() if end_date   else None

    all_folders = sorted(
        [d for d in glob.glob(os.path.join(root_shadowgraph_path, "*")) if os.path.isdir(d)],
        reverse=True,
    )

    if start or end:
        folders, skipped = [], 0
        for folder in all_folders:
            folder_date = _parse_folder_date(folder)
            if folder_date is None:
                logger.warning(f"Cannot parse date from {os.path.basename(folder)}, including anyway.")
                folders.append(folder)
                continue
            if start and folder_date < start:
                skipped += 1; continue
            if end and folder_date > end:
                skipped += 1; continue
            folders.append(folder)
        logger.info(
            f"Date filter {start or 'any'} → {end or 'any'}: "
            f"keeping {len(folders)}, skipping {skipped}"
        )
    else:
        folders = all_folders

    logger.info(
        f"Folders: {len(folders)} / {len(all_folders)}  |  "
        f"folder_workers={n_folder_workers}  cpu_workers_per_folder={n_cpu_workers}"
    )

    args = [
        (folder, save_root, batch_size, window_size, n_cpu_workers)
        for folder in folders
    ]

    with ProcessPoolExecutor(max_workers=n_folder_workers) as executor:
        futures = {executor.submit(_process_one_folder, a): a[0] for a in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Folders"):
            folder, err = future.result()
            if err is not None:
                logger.error(f"Failed: {folder}  →  {err}")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python track_particles.py <input_root> <save_root> "
            "[start_date YYYYMMDD] [end_date YYYYMMDD] "
            "[n_folder_workers] [n_cpu_workers]"
        )
        sys.exit(1)

    root_dir   = sys.argv[1]
    save_root  = sys.argv[2]
    start_date = sys.argv[3] if len(sys.argv) > 3 else None
    end_date   = sys.argv[4] if len(sys.argv) > 4 else None

    # Folder-level parallelism: default 1 (CPU workers saturate all cores per folder)
    n_folder_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    # CPU workers per folder: default = all cores
    n_cpu_workers    = int(sys.argv[6]) if len(sys.argv) > 6 else None

    try:
        process_all_shadowgraph_folders(
            root_shadowgraph_path=root_dir,
            save_root=save_root,
            n_folder_workers=n_folder_workers,
            n_cpu_workers=n_cpu_workers,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Done.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)