import os
import sys
import pandas as pd
import pims
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def draw_vectors_opencv(image, d, scale=5):
    """
    Draw velocity vectors using OpenCV.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB)

    d : DataFrame
        Filtered dataframe for a single frame

    scale : float
        Scaling factor for arrow length
    """

    # Convert grayscale → BGR (needed for colored arrows)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for _, row in d.iterrows():
        x, y = int(row["x"]), int(row["y"])
        dx, dy = row["dx"], row["dy"]

        # Compute endpoint
        end_x = int(x + dx * scale)
        end_y = int(y + dy * scale)  # invert y

        # Compute velocity magnitude
        velocity = np.sqrt(dx**2 + dy**2)

        # Normalize to 0–255 for coloring
        v_norm = min(velocity * 50, 255)

        # Color mapping (blue → red)
        color = (0, int(v_norm), 255 - int(v_norm))

        cv2.arrowedLine(
            image,
            (x, y),
            (end_x, end_y),
            color,
            thickness=2,
            tipLength=0.3
        )

    return image


def process_single_csv(args):
    csv_file, csv_folder, images_parent, output_root = args

    csv_path = os.path.join(csv_folder, csv_file)
    base_name = os.path.splitext(csv_file)[0]
    image_path = os.path.join(images_parent, base_name)

    if not os.path.exists(image_path):
        return

    try:
        data = pd.read_csv(csv_path)
    except:
        return

    if data.empty:
        return

    required_cols = ["frame", "x", "y", "dx", "dy", "area"]
    if not all(col in data.columns for col in required_cols):
        return

    clean_data = data[
        (data["area"] > 100) &
        data[["x", "y", "dx", "dy"]].notna().all(axis=1)
    ]

    grouped = clean_data.groupby("frame")

    try:
        rawframes = pims.ImageSequence(os.path.join(image_path, "*.jpeg"))
    except:
        return

    if len(rawframes) == 0:
        return

    output_dir = os.path.join(output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(rawframes)):
        if i not in grouped.groups:
            continue

        d = grouped.get_group(i)

        frame = rawframes[i]

        # Normalize image to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (255 * (frame / frame.max())).astype(np.uint8)

        # Draw vectors
        frame_out = draw_vectors_opencv(frame, d)

        output_file = os.path.join(
            output_dir,
            f"{base_name}_frame_{i:06d}.png"
        )

        cv2.imwrite(output_file, frame_out)


def main(csv_folder, images_parent, output_root):
    os.makedirs(output_root, exist_ok=True)

    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    args_list = [
        (csv_file, csv_folder, images_parent, output_root)
        for csv_file in csv_files
    ]

    print(f"Using {cpu_count()} CPUs")

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_csv, args_list), total=len(args_list)))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py [csv_folder] [images_parent] [output_folder]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])