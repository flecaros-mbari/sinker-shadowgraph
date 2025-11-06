import cv2
import sys
import os
import uuid
import pims
from loguru import logger
import numpy as np
from multiprocessing.pool import Pool
from functools import partial
from skimage.color import rgb2gray
import glob
import pandas as pd
from tqdm import tqdm
from collections import deque
from PIL import Image
import warnings


# ==============================================================
# BACKGROUND SUBTRACTION
# ==============================================================
def background_subtraction(images, window_size=5, method="median"):
    """
    Generator that yields background-subtracted images using a moving window.

    Parameters
    ----------
    images : iterable of ndarray
        List or generator of grayscale images.
    window_size : int
        Number of frames for background estimation.
    method : str
        'mean' or 'median' for background calculation.

    Yields
    ------
    ndarray : Background-subtracted image.
    """
    buffer = deque(maxlen=window_size)

    for img in tqdm(images):
        buffer.append(img)

        if len(buffer) < window_size:
            # Not enough frames yet
            yield img
        else:
            # Compute background
            stack = np.stack(buffer, axis=0)
            if method == "median":
                bg = np.median(stack, axis=0).astype(np.uint8)
            else:
                bg = np.mean(stack, axis=0).astype(np.uint8)

            # Subtract background
            diff = cv2.absdiff(img, bg)
            yield diff


# ==============================================================
# MULTIPROCESSING HELPER
# ==============================================================
def get_pool(processes):
    """
    Creates multiprocessing pool.

    Parameters
    ----------
    processes : int or "auto"

    Returns
    -------
    pool, map_func
    """
    if processes == "auto":
        processes = None  # Let Pool use all available cores
    elif not isinstance(processes, int):
        raise TypeError("`processes` must be int or 'auto'.")

    if processes is None or processes > 1:
        pool = Pool(processes=processes)
        map_func = pool.imap
    else:
        pool = None
        map_func = map

    return pool, map_func


# ==============================================================
# ROI DETECTION (find_bright_regions_df)
# ==============================================================
def find_bright_regions_df(original_image, binary_mask, save_dir, save=True):
    """
    Finds and characterizes bright regions (contours) in a binary mask.

    Returns
    -------
    pd.DataFrame : Region measurements
    """
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        image_with_contours = original_image.copy()
    else:
        gray = original_image.copy()
        image_with_contours = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []
    roi_dir = os.path.join(save_dir, "ROI")
    contour_dir = os.path.join(save_dir, "contours")

    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        xx = x + w
        yy = y + h
        center_x, center_y = x + w / 2, y + h / 2
        larger = max(w, h)

        h_img, w_img = original_image.shape[:2]
        x1 = max(0, int(center_x - larger / 2))
        y1 = max(0, int(center_y - larger / 2))
        x2 = min(w_img, int(center_x + larger / 2))
        y2 = min(h_img, int(center_y + larger / 2))
        roi = original_image[y1:y2, x1:x2]

        perimeter = cv2.arcLength(cnt, True)
        esd = 2 * np.sqrt(area / np.pi) if area > 0 else 0
        roi_id = str(uuid.uuid4())
        roi_filename = f"{roi_id}_roi_{int(area)}.png"
        roi_path = os.path.join(roi_dir, roi_filename) if save else np.nan

        if save and area > 500 and roi.size > 0:
            success = cv2.imwrite(roi_path, roi)
            if success:
                logger.info(f"Saving ROI into {roi_path}")
            else:
                logger.warning(f"Failed to save ROI at {roi_path}")

        data.append({
            'class': "Unknown",
            'score': 0.1,
            'area': area,
            'saliency': np.nan,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'xx': xx,
            'yy': yy,
            'cluster': -1,
            'perimeter': perimeter,
            'esd': esd,
            'roi_path': roi_path
        })

    if save:
        cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
        contour_image_path = os.path.join(contour_dir, f"{uuid.uuid4()}_contours.png")
        cv2.imwrite(contour_image_path, image_with_contours)
        logger.info(f"Saving contour image to {contour_image_path}")

    return pd.DataFrame(data)


# ==============================================================
# LOCATE: Detect particles in one image
# ==============================================================
def locate(fg_mask, original_image, save=False, save_dir=""):
    """
    Detect particles in one frame using edge detection and contour extraction.
    The contours are detected on the background-subtracted mask,
    but the ROIs are extracted from the original image.
    """
    # 1. Detect edges using the background-subtracted frame
    canny = cv2.Canny(fg_mask, 50, 70, apertureSize=3, L2gradient=True)

    # 2. Dilate to connect edges
    kernel = np.ones((10, 10), np.uint8)
    canny = cv2.dilate(canny, kernel, 2)

    # 3. Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 4. Create binary mask
    mask = np.zeros_like(canny)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # 5. Erode mask to smooth shapes
    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.erode(mask, kernel, 2)

    # 6. Analyze bright regions using the ORIGINAL image
    coords = find_bright_regions_df(original_image, mask, save_dir, save=save)
    return coords


# ==============================================================
# BATCH PROCESSING
# ==============================================================
def batch(original_frames, fg_masks, save_dir, image_paths, save=False, processes='auto', after_locate=None):
    """
    Runs `locate()` on multiple frames (possibly in parallel).
    """
    pool, map_func = get_pool(processes)

    if after_locate is None:
        def after_locate(frame_no, features):
            return features

    try:
        all_features = []
        for i, (fg_mask, original_image) in enumerate(zip(fg_masks, original_frames)):
            image_path = image_paths[i]
            features = locate(fg_mask, original_image, save=save, save_dir=save_dir)

            with Image.open(image_path) as img:
                width, height = img.size

            features['frame'] = i
            features['image_path'] = image_path
            features['image_width'] = width
            features['image_height'] = height

            features = after_locate(i, features)
            logger.info(f"Frame {i}: {len(features)} features")

            if len(features) > 0:
                all_features.append(features)
    finally:
        if pool:
            pool.terminate()

    if len(all_features) > 0:
        return pd.concat(all_features).reset_index(drop=True)
    else:
        warnings.warn("No features found in any frame.")
        return pd.DataFrame(columns=[
            'class', 'score', 'area', 'saliency', 'x', 'y', 'w', 'h',
            'cluster', 'perimeter', 'esd', 'roi_path',
            'frame', 'image_path', 'image_width', 'image_height'
        ])


# ==============================================================
# MAIN FOLDER PROCESSOR
# ==============================================================
def process_shadowgraph_folder(shadowgraph_path, save_root):
    """
    Process all JPEG images in a folder:
    1. Load frames
    2. Subtract background
    3. Detect particles
    4. Save results to CSV
    """
    logger.info(f"Processing folder: {shadowgraph_path}")

    frames = pims.ImageSequence(os.path.join(shadowgraph_path, "*.jpeg"))
    image_paths = sorted(glob.glob(os.path.join(shadowgraph_path, "*.jpeg")))

    if len(frames) == 0:
        logger.warning(f"No images found in {shadowgraph_path}")
        return

    # --- Apply background subtraction to all frames ---
    logger.info("Applying background subtraction...")
    fg_masks = list(background_subtraction(frames, window_size=5, method="median"))

    # --- Detect particles (using fg_mask for detection, original for ROI extraction) ---
    logger.info("Detecting particles...")
    f = batch(frames, fg_masks, image_paths=image_paths, save_dir=save_root, save=True)

    # --- Sanity checks ---
    if 'frame' not in f.columns:
        logger.error("Missing 'frame' column.")
        return

    if not np.issubdtype(f['frame'].dtype, np.integer):
        logger.warning("Converting 'frame' column to int...")
        f['frame'] = f['frame'].astype(int)

    # --- Save results ---
    results_dir = os.path.join(save_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "frames.csv")
    f.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python track_particles.py [input_root] [save_root]")
        sys.exit(1)

    root_dir = sys.argv[1]
    save_root = sys.argv[2]

    try:
        process_shadowgraph_folder(root_dir, save_root)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.exception(f"Error processing {root_dir}: {e}")
