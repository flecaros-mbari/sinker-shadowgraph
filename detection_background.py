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
import cupy as cp


# ==============================================================
# BACKGROUND SUBTRACTION
# ==============================================================
def background_subtraction(images, buffer=None, window_size=5, method="median"):
    """
    GPU-accelerated background subtraction, maintaining continuity with a buffer.
    `buffer` must be a deque or None. Returns (fg_masks, updated_buffer).
    """
    if buffer is None:
        buffer = deque(maxlen=window_size)

    fg_masks = []

    for img in images:
        buffer.append(img)

        if len(buffer) < window_size:
            fg_masks.append(img)  # Not enough background history yet
            continue

        stack = np.stack(buffer, axis=0)

        if method == "median":
            bg_gpu = np.median(stack, axis=0).astype(cp.uint8)
        else:
            bg_gpu = np.mean(stack, axis=0).astype(cp.uint8)

        diff_gpu = cv2.absdiff(img , bg_gpu)
        fg_masks.append(diff_gpu)

    return fg_masks, buffer



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
def find_bright_regions_df(original_image, binary_mask, save_dir, save=True, image_name=None):
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
        if area > 500:
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
        drawn_contours = cv2.drawContours(gray, contours, -1, (0, 0, 255), 2)
        contour_filename = f"{image_name}_contours.png" if image_name else f"{uuid.uuid4()}_contours.png"
        contour_image_path = os.path.join(contour_dir, contour_filename)
        cv2.imwrite(contour_image_path, drawn_contours)
        logger.info(f"Saving contour image to {contour_image_path}")


    return pd.DataFrame(data)


# ==============================================================
# LOCATE: Detect particles in one image
# ==============================================================
def locate(fg_mask, original_image, image_path, save=False, save_dir=""):
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
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    coords = find_bright_regions_df(original_image, mask, save_dir, save=save, image_name=image_name)
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
            features = locate(fg_mask, original_image,image_path, save=save, save_dir=save_dir)

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

def chunks_with_overlap(lst, batch_size, overlap):
    """
    Example: lst=[0..99], batch_size=20, overlap=5
    Output: [0..19], [15..34], [30..49], ...
    """
    step = batch_size - overlap
    for start in range(0, len(lst), step):
        end = min(start + batch_size, len(lst))
        yield lst[start:end]

# ==============================================================
# MAIN FOLDER PROCESSOR
# ==============================================================
def process_shadowgraph_folder(shadowgraph_path, save_root, batch_size=40, window_size=5):
    logger.info(f"Processing folder in batches with background continuity: {shadowgraph_path}")

    image_paths = sorted(glob.glob(os.path.join(shadowgraph_path, "*.jpeg")))
    if len(image_paths) == 0:
        logger.warning(f"No images found in {shadowgraph_path}")
        return

    results_dir = os.path.join(save_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "frames.csv")

    buffer = deque(maxlen=window_size)  # maintains background continuity

    for batch_paths in chunks_with_overlap(image_paths, batch_size, overlap=window_size):

        # Load batch images
        images = [np.array(Image.open(p)) for p in batch_paths]

        logger.info(f"Background subtraction batch: {batch_paths[0]} → {batch_paths[-1]}")
        fg_masks, buffer = background_subtraction(images, buffer=buffer, window_size=window_size)

        logger.info("Detecting particles...")
        features = batch(images, fg_masks, save_dir=save_root, image_paths=batch_paths, save=True)

        if 'frame' in features.columns:
            features['frame'] = features['frame'].astype(int)

        features.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        logger.info(f"Appended {len(features)} detections → {output_path}")

    logger.info("✅ Completed processing all batches with continuous background.")



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
