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
import time
import cupy as cp
import re
from datetime import datetime

def extract_metadata(image_path):
    filename = os.path.basename(image_path)

    # Pattern explanation:
    #   group 1 → instrument name (anything before the first "_")
    #   group 2 → timestamp (e.g., 20250827T015700.321006)
    #   group 3 → camera name (e.g., Shadowgraph)
    #
    # Example filename:
    #   SINKER_20250827T015700.321006Z_Shadowgraph_40297765
    pattern = r"^([^_]+)_(\d{8}T\d{6}\.\d+).*?_(\w+)_"
    m = re.search(pattern, filename)

    if not m:
        return None, None, None, None

    instrument = m.group(1)       # e.g., "SINKER", "ISIIS", etc.
    timestamp_str = m.group(2)    # timestamp string
    camera = m.group(3)           # camera name

    # Convert timestamp string to Python datetime
    clean = timestamp_str.replace("T", "")
    dt = datetime.strptime(clean, "%Y%m%d%H%M%S.%f")

    # Convert to pandas Timestamp
    ts = pd.Timestamp(dt)

    return instrument, timestamp_str, camera, ts


# ==============================================================
# BACKGROUND SUBTRACTION
# ==============================================================


def background_subtraction(images, buffer=None, window_size=5, method="median"):
    """
    Fast GPU background subtraction using CuPy, avoiding CPU-GPU transfers.
    """
    if buffer is None:
        buffer = deque(maxlen=window_size)

    fg_masks = []

    for img_cpu in images:

        # Convert to GPU once
        img = cp.asarray(img_cpu)

        buffer.append(img)

        if len(buffer) < window_size:
            fg_masks.append(img_cpu)
            continue
        
        # Stack remains ON GPU
        stack = cp.stack(list(buffer), axis=0)

        # Compute background ON GPU
        if method == "median":
            bg = cp.median(stack, axis=0)
        else:
            bg = cp.mean(stack, axis=0)

        # Subtraction ON GPU
        diff = cp.abs(img - bg).astype(cp.uint8)

        # Convert back only when returning
        fg_masks.append(cp.asnumpy(diff))

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
def find_bright_regions_df(original_image, binary_mask, save_dir, save=False, image_name=None):
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
    roi_dir = os.path.join(save_dir, "ROIs")
    contour_dir = os.path.join(save_dir, "contours")

    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            xx = x + w
            yy = y + h
            center_x, center_y = (x + xx) / 2, (y + yy) / 2
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

            if save and area > 300 and roi.size > 0:
                success = cv2.imwrite(roi_path, roi)
                if success:
                    logger.info(f"Saving ROI into {roi_path}")
                else:
                    logger.warning(f"Failed to save ROI at {roi_path}")

            data.append({
                'area': area,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'xx': xx,
                'yy': yy,
                'perimeter': perimeter,
                'esd': esd,
                'roi_path': roi_path
            })

    if save:
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]
        drawn_contours = cv2.drawContours(image_with_contours, large_contours, -1, (0, 0, 255), 2)
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
def batch(original_frames, fg_masks, save_dir, image_paths, index, save=False, processes='auto', after_locate=None):
    pool, map_func = get_pool(processes)

    if after_locate is None:
        def after_locate(frame_no, features):
            return features

    try:
        all_features = []
        for i, (fg_mask, original_image) in enumerate(zip(fg_masks, original_frames)):
            image_path = image_paths[i]
            
            # Run your locate()
            features = locate(fg_mask, original_image, image_path, save=save, save_dir=save_dir)

            # Extract timestamp + camera
            intrument, timestamp_str, camera, dt = extract_metadata(image_path)

            # Get width and height
            with Image.open(image_path) as img:
                width, height = img.size

            # Add metadata
                
            features["frame"] = index + i
            features["image_path"] = image_path
            features["image_width"] = width
            features["image_height"] = height
            features["timestamp"] = dt
            features["camera"] = camera
            features["instrument"] = intrument
            features['class']= "unknown"
            features['label']= "unknown"
            features['model']= "unknown"


            # Create empty new columns
            features["particle_id"] = np.nan
            features["dx"] = np.nan
            features["dy"] = np.nan
            features["speed"] = np.nan

            logger.info(f"Frame {i}: {len(features)} features (camera={camera}, time={dt}), timestamp={timestamp_str}")

            # Apply custom hook
            features = after_locate(i, features)

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
            'frame', 'image_path', 'image_width', 'image_height',
            'timestamp', 'camera'
        ])


def process_shadowgraph_folder(shadowgraph_path, save_root, batch_size=40, window_size=3):
    logger.info(f"Processing folder with STATIC background window: {shadowgraph_path}")
    start_total = time.time()


    image_paths = sorted(glob.glob(os.path.join(shadowgraph_path, "*.jpeg")))
    if len(image_paths) < window_size:
        logger.error(f"Not enough images to create static window ({window_size} required).")
        return

    results_dir = os.path.join(save_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "frames.csv")

    static_window_paths = image_paths[:window_size]
    static_window_images = [np.array(Image.open(p)) for p in static_window_paths]

    logger.info(f"Static background window created from: {static_window_paths[0]} → {static_window_paths[-1]}")

    static = time.time()
    # Llamada especial a tu background_subtraction para crear el modelo inicial
    fg_masks_static, static_buffer = background_subtraction(
        static_window_images,
        buffer=None,
        window_size=window_size
    )
    staticend = time.time()
    logger.info(f"Static: {staticend-static} sec")


    # YA NO ACTUALIZAMOS static_buffer — queda congelado aquí.
    # static_buffer ahora es tu background fijo para todo el procesamiento.

    # ======================================================
    # 📌 Procesar en batches normales, sin overlap y sin buffer dinámico
    # ======================================================
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]

        images = [np.array(Image.open(p)) for p in batch_paths]

        logger.info(f"Background subtraction batch: {batch_paths[0]} → {batch_paths[-1]}")

        # Usar SIEMPRE el mismo buffer estático
        back = time.time()
        fg_masks, _ = background_subtraction(
            images,
            buffer=static_buffer,
            window_size=window_size
        )
        backend = time.time()
        logger.info(f"Background subtraction batch: {backend-back} sec")

        logger.info("Detecting particles...")
        featuress = time.time()
        features = batch(
            images,
            fg_masks,
            save_dir=save_root,
            image_paths=batch_paths,
            index = i,
            save=True
        )
        featuresend= time.time()
        logger.info(f"Features: {featuresend - featuress} sec")
        if 'frame' in features.columns:
            features['frame'] = features['frame'].astype(int)

        csv = time.time()
        features.to_csv(
            output_path,
            mode='a',
            header=not os.path.exists(output_path),
            index=False
        )
        csvend = time.time()
        logger.info(f"Csv: {csvend - csv} sec")

        logger.info(f"Appended {len(features)} detections → {output_path}")
    
    end_total = time.time()
    total_time = end_total - start_total
    logger.info(f"⏳ Total processing time of : {total_time:.2f} seconds ({total_time/60:.2f} minutes) for {len(image_paths)} images")
    logger.info(f"Time per image {total_time/ len(image_paths)} ")


    logger.info("✅ Completed processing with STATIC background.")


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
