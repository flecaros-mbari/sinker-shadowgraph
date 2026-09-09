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
    instrument    = m.group(1)
    timestamp_str = m.group(2)
    camera        = m.group(3)
    clean = timestamp_str.replace("T", "")
    dt    = datetime.strptime(clean, "%Y%m%d%H%M%S.%f")
    ts    = pd.Timestamp(dt)
    wipe  = is_even_hour(ts)
    return instrument, timestamp_str, camera, ts, wipe


# ==============================================================
# IMAGE LOADING  (parallelized)
# ==============================================================

def _load_image(p):
    return np.array(Image.open(p))


def load_images_parallel(paths, n_workers=None):
    n_workers = n_workers or min(32, len(paths))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_load_image, p) for p in paths]
        return [f.result() for f in futures]


# ==============================================================
# BACKGROUND SUBTRACTION  (GPU)
# ==============================================================

def compute_rolling_background(all_paths, batch_start, window_size, n_workers=None):
    half = window_size // 2
    lo   = max(0, batch_start - half)
    hi   = min(len(all_paths), batch_start + half + (window_size % 2))
    window_paths = all_paths[lo:hi]

    pool       = cp.get_default_memory_pool()
    gpu_frames = []
    try:
        for p in window_paths:
            img_cpu = _load_image(p)
            gpu_frames.append(cp.asarray(img_cpu))
            del img_cpu
        stack          = cp.stack(gpu_frames, axis=0)
        del gpu_frames
        background_gpu = cp.median(stack, axis=0).astype(stack.dtype)
        del stack
        background_cpu = cp.asnumpy(background_gpu)
        del background_gpu
    finally:
        pool.free_all_blocks()
    return background_cpu


def _subtract_one_frame_cpu(img_cpu, static_bg_cpu):
    return np.abs(
        img_cpu.astype(np.int16) - static_bg_cpu.astype(np.int16)
    ).clip(0, 255).astype(np.uint8)


def background_subtraction_batch(images, static_bg_cpu, gpu_chunk_size=4):
    pool    = cp.get_default_memory_pool()
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
            logger.warning(f"GPU OOM on chunk of {len(chunk)}, retrying frame-by-frame")
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
# PARTICLE DETECTION  (replaces saliency pyramid)
# ==============================================================

def particle_mask(original_image: np.ndarray, fg_diff: np.ndarray,
                  diff_threshold: int = 7) -> np.ndarray:
    """
    Detect particles as the intersection of two independent masks.

    Mask A — background-subtraction diff mask
        Pixels where |frame - rolling_background| > diff_threshold.
        Shadowgraph particles produce a diff of ~8-25 DN against the median
        background; sensor noise stays below ~7 DN.  A fixed low threshold
        works reliably here because the rolling median already removes all
        static structure — anything that exceeds the threshold genuinely moved.

    Mask B — adaptive threshold on the original frame
        cv2.adaptiveThreshold with a Gaussian neighbourhood of 31 px and
        constant C=3 marks pixels that are darker than their local background.
        Shadowgraph particles are always darker than the surrounding grey field,
        so this is a very stable discriminator for this sensor type.
        A 2×2 morphological open removes single-pixel speckle; a 2×2 dilation
        closes hairline gaps within individual particle bodies.

    Why not the saliency pyramid?
        The HSV/LAB saliency pyramid was designed for natural colour images.
        Shadowgraph frames are near-uniform grey with tiny dark specks — there
        is almost no hue or saturation variation to exploit — so the saliency
        threshold was either too loose (v1: passed 93 % of pixels → one giant
        blob) or too tight (v2: passed almost nothing → missed everything).
        Adaptive thresholding on the original frame is far more robust here
        because "darker than local neighbourhood" is exactly what a shadowgraph
        particle is by definition.

    Signature note
        This function takes numpy arrays, NOT file paths.  The original
        fine_grained_saliency_pyramid(image_path: str) accepted a file path
        and called cv2.imread() internally; passing a pre-loaded array to it
        caused the "Expected 'filename' to be a str" OpenCV error seen in
        production.  All callers already have the image in memory, so there
        is no need to touch disk again.

    Parameters
    ----------
    original_image : np.ndarray  uint8, BGR or grayscale, the raw camera frame
    fg_diff        : np.ndarray  uint8, abs-difference image from bg subtraction
    diff_threshold : int         Minimum diff DN to be considered foreground (default 7)

    Returns
    -------
    np.ndarray  uint8 binary mask — 255 = particle, 0 = background
    """
    # -- Grayscale conversions -----------------------------------------
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_image

    if len(fg_diff.shape) == 3:
        diff_gray = cv2.cvtColor(fg_diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = fg_diff

    # -- Mask A: anything that changed from the rolling background ------
    diff_mask = (diff_gray > diff_threshold).astype(np.uint8) * 255

    # -- Mask B: pixels darker than their local 31-px neighbourhood -----
    # blockSize=31 spans the largest expected particles (~20 px diameter)
    # while still adapting to gentle illumination gradients across the frame.
    # C=3 sets how much darker than the local mean a pixel must be.
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=3,
    )
    # Remove single-pixel speckle
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Close tiny gaps within particle bodies
    adaptive = cv2.dilate(adaptive, np.ones((2, 2), np.uint8), iterations=1)

    # -- Intersection: foreground AND dark = particle -------------------
    return cv2.bitwise_and(diff_mask, adaptive)


# ==============================================================
# ROI DETECTION
# ==============================================================

def find_bright_regions_df(original_image, binary_mask, save_dir,
                            save=True, image_name=None,
                            min_area=80, max_area=500_000):
    if len(original_image.shape) == 3:
        image_with_contours = original_image.copy()
    else:
        image_with_contours = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h   = cv2.boundingRect(cnt)
        xx, yy       = x + w, y + h
        center_x     = (x + xx) / 2
        center_y     = (y + yy) / 2
        larger       = max(w, h)
        h_img, w_img = original_image.shape[:2]
        x1 = max(0,     int(center_x - larger / 2))
        y1 = max(0,     int(center_y - larger / 2))
        x2 = min(w_img, int(center_x + larger / 2))
        y2 = min(h_img, int(center_y + larger / 2))
        roi       = original_image[y1:y2, x1:x2]
        perimeter = cv2.arcLength(cnt, True)
        esd       = 2 * np.sqrt(area / np.pi)
        roi_id    = str(uuid.uuid4())

        roi_dir  = (os.path.join(save_dir, "ROIs", image_name)
                    if image_name else os.path.join(save_dir, "ROIs"))
        roi_path = (os.path.join(roi_dir, f"{roi_id}_roi_{int(area)}.png")
                    if save else np.nan)

        if save and roi.size > 0:
            os.makedirs(roi_dir, exist_ok=True)
            if not cv2.imwrite(roi_path, roi):
                logger.warning(f"Failed to save ROI at {roi_path}")

        data.append({
            'detection_id': roi_id,
            'area':         area,
            'x': x, 'y': y, 'w': w, 'h': h,
            'xx': xx, 'yy': yy,
            'perimeter':    perimeter,
            'esd':          esd,
            'roi_path':     roi_path,
        })

    if save:
        contour_dir = os.path.join(save_dir, "contours")
        os.makedirs(contour_dir, exist_ok=True)
        valid_contours   = [c for c in contours
                            if min_area <= cv2.contourArea(c)]
        drawn            = cv2.drawContours(image_with_contours, valid_contours, -1, (0, 0, 255), 2)
        contour_filename = (f"{image_name}_contours.png"
                            if image_name else f"{uuid.uuid4()}_contours.png")
        cv2.imwrite(os.path.join(contour_dir, contour_filename), drawn)

    return pd.DataFrame(data)


def locate(combined_mask, original_image, image_path, save=True, save_dir=""):
    """
    Fill particle blobs from the combined particle mask and return ROI records.

    Morphology is intentionally minimal — particle_mask() has already done
    the heavy lifting.  We just close hairline gaps within a single particle
    body before filling contours.
    """
    canny = cv2.Canny(combined_mask, 130, 150, apertureSize=3, L2gradient=True)
    canny = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=3)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(canny)
    cv2.drawContours(mask, contours, -1, 255, -1)
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return find_bright_regions_df(original_image, mask, save_dir,
                                  save=save, image_name=image_name)


# ==============================================================
# FRAME WORKER
# ==============================================================

_EMPTY_ROW = {
    'detection_id': None, 'area': None,
    'x': None, 'y': None, 'w': None, 'h': None,
    'xx': None, 'yy': None,
    'perimeter': None, 'esd': None, 'roi_path': None,
}


def _locate_one(fg_mask, original_image, image_path, save, save_dir, frame_index):
    """
    Process one frame end-to-end.

    NOTE: fg_mask and original_image are numpy arrays (already loaded).
    particle_mask() accepts arrays directly — do NOT pass file paths here.
    The original bug was fine_grained_saliency_pyramid(image_path: str)
    being called with an array, causing the OpenCV imread type error.
    """
    instrument, _, camera, dt, wipe = extract_metadata(image_path)
    height, width = original_image.shape[:2]

    if wipe:
        features = pd.DataFrame([_EMPTY_ROW.copy()])
    else:
        # particle_mask takes arrays, not paths — no cv2.imread() inside
        combined = particle_mask(original_image, fg_mask)
        features = locate(combined, original_image, image_path,
                          save=save, save_dir=save_dir)
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
# BATCH
# ==============================================================

def batch(original_frames, fg_masks, save_dir, image_paths, index,
          save=True, n_workers=None):
    n_workers = n_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_locate_one, fg_masks[i], original_frames[i],
                      image_paths[i], save, save_dir, index + i)
            for i in range(len(original_frames))
        ]
        frame_dfs = [f.result() for f in futures]

    if frame_dfs:
        result          = pd.concat(frame_dfs, ignore_index=True)
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

def process_shadowgraph_folder(shadowgraph_path, save_root,
                                batch_size=50, window_size=10,
                                n_cpu_workers=None):
    n_cpu_workers = n_cpu_workers or os.cpu_count()
    logger.info(f"Processing: {shadowgraph_path}  (cpu_workers={n_cpu_workers})")
    start_total = time.time()

    image_paths = sorted(glob.glob(os.path.join(shadowgraph_path, "*.jpeg")))
    if not image_paths:
        logger.error("No JPEG images found.")
        return

    folder_name = os.path.basename(shadowgraph_path)
    logger.info(f"Folder: {folder_name}")

    if len(image_paths) < window_size:
        logger.error(f"Need at least {window_size} images for background window.")
        return

    output_path     = os.path.join(save_root, f"{folder_name}.csv")
    output_time_txt = os.path.join(save_root, f"{folder_name}.txt")

    if os.path.exists(output_path) and os.path.exists(output_time_txt):
        logger.info(f"Skipping {folder_name}: outputs already exist.")
        return
    if os.path.exists(output_path) and not os.path.exists(output_time_txt):
        logger.warning(f"Incomplete outputs for {folder_name}, reprocessing.")
        os.remove(output_path)

    timestamps = []
    for p in image_paths:
        _, _, _, ts, _ = extract_metadata(p)
        if ts is not None:
            timestamps.append(ts)

    if len(timestamps) > 1:
        ts_sorted   = sorted(timestamps)
        diffs       = [(ts_sorted[i+1]-ts_sorted[i]).total_seconds()
                       for i in range(len(ts_sorted)-1)]
        median_diff = np.median(diffs)
        frame_rate  = 1.0 / median_diff if median_diff > 0 else None
        logger.info(f"Frame rate: {frame_rate:.3f} Hz")
    else:
        frame_rate = None
        logger.warning("Not enough timestamps to compute frame rate.")

    all_batch_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc=folder_name):
        batch_paths = image_paths[i: i + batch_size]

        t0     = time.time()
        images = load_images_parallel(batch_paths, n_workers=n_cpu_workers)
        logger.info(f"Image loading:          {time.time()-t0:.2f}s  ({len(batch_paths)} frames)")

        t0         = time.time()
        rolling_bg = compute_rolling_background(image_paths, i, window_size, n_cpu_workers)
        logger.info(f"Rolling background:     {time.time()-t0:.2f}s  "
                    f"(window centred on frame {i}, size {window_size})")

        t0       = time.time()
        fg_masks = background_subtraction_batch(images, rolling_bg)
        logger.info(f"Background subtraction: {time.time()-t0:.2f}s")

        t0       = time.time()
        features = batch(images, fg_masks, save_root, batch_paths, i,
                         save=True, n_workers=n_cpu_workers)
        logger.info(f"Feature extraction:     {time.time()-t0:.2f}s")

        features["frame_rate"] = frame_rate
        all_batch_features.append(features)

    os.makedirs(save_root, exist_ok=True)
    full_df = pd.concat(all_batch_features, ignore_index=True)
    full_df.to_csv(output_path, index=False)
    logger.info(f"CSV written: {output_path}  ({len(full_df)} rows)")

    total_time = time.time() - start_total
    logger.info(f"Total time: {total_time:.1f}s  ({total_time/len(image_paths):.4f}s/image)")

    with open(output_time_txt, "w") as f:
        f.write(f"Folder: {folder_name}\n")
        f.write(f"Images processed: {len(image_paths)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"CPU workers: {n_cpu_workers}\n")
        f.write(f"Frame rate (Hz): {frame_rate:.3f}\n" if frame_rate
                else "Frame rate (Hz): unknown\n")
        f.write(f"Total time (seconds): {total_time:.2f}\n")
        f.write(f"Total time (minutes): {total_time/60:.2f}\n")
        f.write(f"Time per image (seconds): {total_time/len(image_paths):.4f}\n")


# ==============================================================
# FOLDER WORKER / DATE FILTER / MAIN ENTRY
# ==============================================================

def _process_one_folder(args):
    folder, save_root, batch_size, window_size, n_cpu_workers = args
    try:
        process_shadowgraph_folder(folder, save_root, batch_size, window_size, n_cpu_workers)
        return folder, None
    except Exception as e:
        logger.exception(f"Error in folder {folder}: {e}")
        return folder, e


def _parse_folder_date(folder_path):
    name = os.path.basename(folder_path)
    m = re.search(r"(\d{8})", name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
    return None


def process_all_shadowgraph_folders(root_shadowgraph_path, save_root,
                                     batch_size=30, window_size=10,
                                     n_folder_workers=1, n_cpu_workers=None,
                                     start_date=None, end_date=None):
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
            fd = _parse_folder_date(folder)
            if fd is None:
                folders.append(folder); continue
            if start and fd < start: skipped += 1; continue
            if end   and fd > end:   skipped += 1; continue
            folders.append(folder)
        logger.info(f"Date filter: keeping {len(folders)}, skipping {skipped}")
    else:
        folders = all_folders

    logger.info(
        f"Folders: {len(folders)} / {len(all_folders)}  |  "
        f"folder_workers={n_folder_workers}  cpu_workers_per_folder={n_cpu_workers}"
    )

    args = [(f, save_root, batch_size, window_size, n_cpu_workers) for f in folders]
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
            "Usage: python detection_background_v4.py <input_root> <save_root> "
            "[start_date YYYYMMDD] [end_date YYYYMMDD] "
            "[n_folder_workers] [n_cpu_workers]"
        )
        sys.exit(1)

    root_dir   = sys.argv[1]
    save_root  = sys.argv[2]
    start_date = sys.argv[3] if len(sys.argv) > 3 else None
    end_date   = sys.argv[4] if len(sys.argv) > 4 else None
    n_folder_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
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