# vggt-eval

Evaluation and extension suite for **VGGT** (Visual Geometry Grounded Transformer, Meta AI / Oxford VGG, CVPR 2025).

> **Hardware target:** Kaggle / Colab free-tier T4 GPU (14.6 GB VRAM)  
> **Dataset:** TUM Visual-Inertial (EuRoC format)

---

## Quick start (Kaggle / Colab)

Every notebook is self-contained. Run the **Setup cell (Cell 0)** first — it installs dependencies, clones VGGT and this repo, then restarts the kernel. After the restart, run all remaining cells in order.

```python
# The setup cell does this automatically:
pip install --upgrade numpy
pip install plotly pandas pillow scipy matplotlib
git clone https://github.com/facebookresearch/vggt.git && pip install -e vggt
git clone https://github.com/insha-py/vggt-eval.git
```

> **Kaggle note:** The kernel restart after `pip install --upgrade numpy` is mandatory to avoid a numpy ABI mismatch. The setup cell handles this automatically.

---

## Notebooks

| Notebook | What it does | Runs on |
|---|---|---|
| `00_results_dashboard.ipynb` | Visualise all experiment results — no inference needed | Local |
| `01_pipeline_basic.ipynb` | End-to-end VGGT inference: poses, depth maps, point cloud, COLMAP export | Kaggle |
| `02_sequential_processing.ipynb` | Long-sequence processing via sliding windows + Procrustes alignment | Kaggle |
| `03_imu_integration.ipynb` | SLERP-blend VGGT poses with IMU gyro integration, alpha sweep | Kaggle |
| `04b_resolution_single.ipynb` | Resolution sweep 224–518 px, single-pass (8 frames) | Kaggle |
| `06_multisequence.ipynb` | Resolution sweep replicated across all 4 TUM-VI sequences | Kaggle |
| `07_metric_scale.ipynb` | Umeyama scale factor analysis across sequences and resolutions | Kaggle |
| `08_imu_frame_selection.ipynb` | IMU-guided frame selection vs uniform baseline | Kaggle |

### Results dashboard (local)

`00_results_dashboard.ipynb` loads from `results/` CSVs and renders all plots with Plotly. No GPU required.

```bash
cd vggt-eval
jupyter notebook notebooks/00_results_dashboard.ipynb
```

---

## Dataset: TUM Visual-Inertial

Sequences used: `room1`, `room2`, `corridor1`, `slides1` (EuRoC format, 512 px export).

The downloader fetches only the files needed (~35 MB per sequence via HTTP Range requests — not the full 1.6 GB archive).

```python
from src.tum_vi import TUMVIDataset

ds = TUMVIDataset(sequence="room1", n_frames=24, download_dir="/tmp/tumvi")
ds.download()
data = ds.load()

# data keys:
#   image_paths       list[str]       paths to extracted frames
#   image_timestamps  list[float]     seconds
#   imu_readings      list[IMUReading] 200 Hz gyro + accel
#   gt_extrinsics     np.ndarray (N, 3, 4)  motion-capture GT
#   calib             dict            camera intrinsics + IMU-camera transform
```

Available sequences: `room1`, `room2`, `corridor1`, `slides1`, `magistrale1`, `outdoors1`.

---

## Source library (`src/`)

### `pipeline.py` — VGGT inference

```python
from src.pipeline import VGGTPipeline, load_images_from_list, run_vggt_inference

pipe = VGGTPipeline()
pipe.load_model()                          # downloads weights on first run

imgs, _ = load_images_from_list(paths, target_size=224)
out = run_vggt_inference(pipe.model, imgs, pipe.device, pipe.dtype, resolution=224)

# out keys: extrinsic (N,3,4), intrinsic (N,3,3), depth (N,H,W), point_map (N,H,W,3)
```

### `metrics.py` — ATE and RPE

```python
from src.metrics import compute_ate, compute_rpe

ate = compute_ate(pred_extrinsics, gt_extrinsics, align=True, with_scale=True)
# ate keys: mean, rmse, median, scale_factor

rpe = compute_rpe(pred_extrinsics, gt_extrinsics, step=1)
# rpe keys: trans_mean, trans_rmse, rot_mean_deg
```

### `imu.py` — IMU processing and frame selection

```python
from src.imu import IMUPreintegrator, estimate_gyro_bias, select_frames_by_rotation

# Estimate and remove gyro bias from the first second of data
bias = estimate_gyro_bias(imu_readings, duration_s=1.0)

# Select frames where camera has rotated >= theta degrees since last accepted frame
indices = select_frames_by_rotation(
    imu_readings,
    image_timestamps,
    theta_min_deg=5.0,
    gyro_bias=bias,
    max_frames=8,          # hard cap for VRAM safety
)
# returns sorted list of indices into image_timestamps
```

### `resolution_sweep.py` — Resolution sweep

```python
from src.resolution_sweep import ResolutionSweeper

sweeper = ResolutionSweeper(pipe, resolutions=[224, 336, 448, 518])
results = sweeper.run(image_paths, gt_extrinsics)
# results: list[ResolutionResult] with fields: resolution, ate, rpe, time_s, peak_mb, scale_factor
```

### `chunking.py` — Sliding window for long sequences

```python
from src.chunking import SlidingWindowProcessor

proc = SlidingWindowProcessor(pipe, window_size=8, overlap=3)
poses = proc.process(image_paths)          # aligned global trajectory
```

### `imu_fusion.py` — SLERP pose blending

```python
from src.imu_fusion import IMUVGGTFusion

fusion = IMUVGGTFusion(calib)
result = fusion.alpha_sweep(
    vggt_extrinsics, imu_readings, image_timestamps,
    alphas=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    gt_extrinsics=gt_extrinsics,
)
# result: DataFrame with columns alpha, ate_mean, ate_rmse
```

---

## Results files

Experiment outputs are saved to `results/` as CSVs and loaded by the dashboard.

| File | Content |
|---|---|
| `results/phase7_scale_across_sequences.csv` | Umeyama scale factor per sequence × resolution |
| `results/phase8_imu_frame_selection.csv` | IMU selection sweep, room1 |
| `results/phase8_multisequence.csv` | IMU selection sweep, all 4 sequences |

---

## Safe frame budget (T4 GPU, 14.6 GB VRAM)

| Resolution | Max safe frames |
|---|---|
| 518 px | 8 |
| 448 px | 10 |
| 336 px | 14 |
| 224 px | 20+ |

Always set `MAX_VGGT_FRAMES = 8` when sweeping at 518 px to avoid OOM.

---

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0 + CUDA
- `numpy`, `scipy`, `pandas`, `pillow`, `plotly`, `matplotlib`
- `vggt` (auto-cloned by setup cells)

---

## Citation

```bibtex
@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and
               Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {CVPR},
  year      = {2025}
}
```
