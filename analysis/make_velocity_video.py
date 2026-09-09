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
            # t = tp.filter_stubs(t, 2)
            
            # for item in set(t.particle):
            #     logger.info(f"Filtering particle: {item}")
            #     sub = t[t.particle == item]

            #     # calcular ventana mínima
            #     win = min(5, len(sub))
            #     if win % 2 == 0:  # asegurar impar
            #         win -= 1
            #     if win > 3:  # aplicar filtro solo si mayor que polyorder
            #         t.loc[t.particle == item, 'x'] = savgol_filter(sub.x, window_length=win, polyorder=3)
            #         t.loc[t.particle == item, 'y'] = savgol_filter(sub.y, window_length=win, polyorder=3)

            # t.to_csv(filtered_tracks_path, index=False)

        else:  # este else pertenece al `elif os.path.exists(tracks_path):`
            logger.error("No tracks file found in results folder.")
            exit()


        data = pd.DataFrame()
        rows = []
        for particle in t.particle_id.unique():
            sub = t[t.particle_id == particle].sort_values('frame')
            dvx = np.diff(sub.x.values)
            dvy = np.diff(sub.y.values)
            areas = sub.area.values
            for x, y, dx, dy, frame, a in zip(sub.x.values[:-1], sub.y.values[:-1], dvx, dvy, sub.frame.values[:-1], areas[:-1]):
                rows.append({'dx': dx, 'dy': dy, 'x': x, 'y': y, 'frame': frame, 'particle_id': particle, 'area': a})
                logger.info('dx ', dx, ' dy ', dy, ' x ', x, ' y ', y, ' frame ', frame, ' particle_id', particle, ' area ', a)
        data = pd.DataFrame(rows)


        data.to_csv(velocities_path, index=False)
    else:
        data = pd.read_csv(velocities_path)

    if not os.path.exists(image_path):
        logger.error(f"Image path not found: {image_path}")
        exit()

    rawframes = pims.ImageSequence(os.path.join(image_path, "*.jpeg"))
    output_dir = os.path.join(base_path, 'tracks')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(rawframes)):
        logger.info(f"\n--- Frame {i} ---")
        d = data[(data.frame == i) & (data.area > 300)]
        logger.info(f"rawframe shape: {rawframes[i].shape}")
        logger.info(f"Particles in this frame: {len(d)}")
        logger.info(f"dx stats: min={d.dx.min() if len(d)>0 else None}, max={d.dx.max() if len(d)>0 else None}")

        
        fig, ax = plt.subplots()
        plt.imshow(rawframes[i], cmap='gray', vmin=0, vmax=255)
        colormap = cm.viridis
        colors = 24.0 * 3600 * 10 * np.sqrt(d.dx**2 + d.dy**2) / 1_000_000  # m/day
        norm = Normalize(vmin=0, vmax=255)
        plt.quiver(d.x, d.y, d.dx, -d.dy, color=colormap(norm(colors)), scale_units='xy', scale=0.1, pivot='tail', width=0.0008, headwidth=5, headlength=5)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
        cbar.set_label('Estimated speed (m/d)')
        plt.savefig(os.path.join(output_dir, f"viz_particles_frame_{i:06d}.png"), dpi=300)
        plt.close(fig)
