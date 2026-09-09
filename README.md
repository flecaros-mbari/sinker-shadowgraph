[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/downloads/)

**sinker-shadowgraph**

Particle detection, tracking, and velocity/size analysis for the SINKER mooring's
Shadowgraph camera (instrument `Shadowgraph_40297765`, MARS mooring). Given a folder
of raw frames, the pipeline detects particles, tracks them across frames with a
Kalman filter, and produces a combined detections+tracking CSV plus overview plots.

**Repository layout**

- `detection/` — `old_detector.py` (background-diff mask; validated against real
  Shadowgraph frames and the detector the pipeline actually uses) and
  `detection_background.py` (adds an adaptive-threshold mask that over-triggers on
  this instrument's background texture — kept for reference/future retuning, not
  used by `pipeline.py`)
- `tracking/` — `tracking_improved.py` (current Kalman-filter tracker) and
  `test_tracking.py` (earlier tracker, superseded)
- `analysis/` — `plot.py` (overview + ESD-violin plots, used by `pipeline.py`),
  `checking_kalman.py` (QA/diagnostic tool for inspecting individual tracks),
  `csv_combination.py` (merge per-folder detection CSVs), `fernanda-approach.py`,
  `make_velocity_video.py`
- `syntetic/` — `testing-blener.py`, synthetic-frame detection tests
- `pipeline.py` — end-to-end entry point (detect → track → plot) for one
  Shadowgraph hour-folder
- `run_pipeline_hourly.sh` — wrapper invoked by the `sinker-pipeline` systemd timer;
  finds the latest complete hour-folder under `/mnt/SINKER/MARS/Shadowgraph_40297765`
  and runs `pipeline.py` on it

**Installation**

```
conda create -n sinker python=3.13
conda activate sinker
pip install -r requirements.txt
```

**Running the pipeline manually**

```
python pipeline.py <images_dir> [output_dir] [options]
```

- `images_dir` — folder of raw `*.jpeg` frames (e.g. a `Shadowgraph_<id>/<year>/<month>/<day>/<hour>` folder)
- `output_dir` — optional; if omitted, derived from `images_dir` under `/mnt/Durkin_Data/SINKER_processed/...`
- `--batch-size`, `--window-size` — detection background-subtraction batching
- `--cpu-workers` — parallel CPU workers for image loading
- `--viz` — also render annotated frames + a tracking video (off by default)
- `--no-plots` — skip overview/ESD plots
- `--save-rois` — save per-particle ROI crops + contour overlays during detection
- `--min-esd-mm` — minimum ESD (mm) kept for plotting (detection/tracking still run on everything)
- `--fps` — frame rate used for the tracking video

**Automated hourly run**

A systemd user timer (`sinker-pipeline.timer` / `sinker-pipeline.service`) runs
`run_pipeline_hourly.sh` shortly after each hour rolls over, processing the most
recent *complete* hour-folder found under `/mnt/SINKER/MARS/Shadowgraph_40297765`
(skipping the newest folder, which may still be receiving frames). Already-processed
folders are tracked in `~/.local/state/sinker-pipeline/last_processed_folder` and
skipped on subsequent runs.
