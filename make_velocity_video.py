import os
import sys
import cv2
import pims
from loguru import logger
import numpy as np
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

if __name__ == "__main__":

    if len(sys.argv) < 3:
        logger.error("Usage: python make_velocity_video.py [base_path_with_results] [image_path_with_jpegs]")
        exit()

    base_path = sys.argv[1]
    image_path = sys.argv[2]
    results_path = os.path.join(base_path, "results")

    if not os.path.exists(results_path):
        logger.error(f"No 'results' folder found at: {results_path}")
        exit()

    filtered_tracks_path = os.path.join(results_path, "filtered_tracks.csv")
    tracks_path = os.path.join(results_path, "sinker_tracked.csv")
    velocities_path = os.path.join(results_path, "velocities.csv")

    if True:
        if os.path.exists(filtered_tracks_path):
            t = pd.read_csv(filtered_tracks_path)
        elif os.path.exists(tracks_path):
            t = pd.read_csv(tracks_path)
            t = tp.filter_stubs(t, 2)

            for item in set(t.particle):
                logger.info(f"Filtering particle: {item}")
                sub = t[t.particle == item]

                # calcular ventana mínima
                win = min(5, len(sub))
                if win % 2 == 0:  # asegurar impar
                    win -= 1
                if win > 3:  # aplicar filtro solo si mayor que polyorder
                    t.loc[t.particle == item, 'x'] = savgol_filter(sub.x, window_length=win, polyorder=3)
                    t.loc[t.particle == item, 'y'] = savgol_filter(sub.y, window_length=win, polyorder=3)

            t.to_csv(filtered_tracks_path, index=False)

        else:  # este else pertenece al `elif os.path.exists(tracks_path):`
            logger.error("No tracks file found in results folder.")
            exit()


        data = pd.DataFrame()
        for item in set(t.particle):
            sub = t[t.particle == item]
            dvx = np.diff(sub.x)
            dvy = np.diff(sub.y)
            area = sub.area
            for x, y, dx, dy, frame, area in zip(sub.x[:-1], sub.y[:-1], dvx, dvy, sub.frame[:-1], area):
                logger.info(f"frame: {frame}, particle: {item}")
                data = pd.concat([data, pd.DataFrame([{
                    'dx': dx, 'dy': dy, 'x': x, 'y': y, 'frame': frame, 'particle': item, 'area': area
                }])], axis=0)

        data.to_csv(velocities_path, index=False)
    else:
        data = pd.read_csv(velocities_path)

    if not os.path.exists(image_path):
        logger.error(f"Image path not found: {image_path}")
        exit()

    rawframes = pims.ImageSequence(os.path.join(image_path, "*.jpeg"))
    output_dir = os.path.join(base_path, 'output_frames_v2')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(rawframes)):
        logger.info(f"Visualizing frame: {i}")
        d = data[(data.frame == i) & (data.area > 100)]
        fig, ax = plt.subplots()
        plt.imshow(rawframes[i], cmap='gray', vmin=0, vmax=255)
        colormap = cm.viridis
        colors = 24.0 * 3600 * 10 * np.sqrt(d.dx**2 + d.dy**2) / 1_000_000  # m/day
        norm = Normalize(vmin=0, vmax=25)
        plt.quiver(d.x, d.y, d.dx, -d.dy, color=colormap(norm(colors)), scale_units='xy', scale=0.1, pivot='tail', width=0.0008, headwidth=5, headlength=5)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        cbar.set_label('Estimated speed (m/d)')
        plt.savefig(os.path.join(output_dir, f"viz_particles_frame_{i:06d}.png"), dpi=300)
        plt.close(fig)
