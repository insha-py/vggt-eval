# vggt-eval

Undergraduate thesis project evaluating **VGGT** (Visual Geometry Grounded Transformer, Meta AI) for 3D reconstruction, with extensions for sliding-window processing, IMU fusion, and resolution sensitivity analysis.

> **Author:** Insha Khan  
> **Hardware target:** Colab / Kaggle free-tier T4 GPU (16 GB VRAM)

---

## What this repo achieves

| Phase | Notebook | What it does |
|-------|----------|--------------|
| **2 — Basic pipeline** | `notebooks/01_pipeline_basic.ipynb` | End-to-end VGGT inference, point cloud, COLMAP export, benchmark |
| **3 — Sliding window** | `notebooks/02_sequential_processing.ipynb` | Long-sequence processing via sliding windows + Procrustes chunk alignment |
| **4 — IMU integration** | `notebooks/03_imu_integration.ipynb` | Gyro pre-integration, SLERP-blend with VGGT poses, ATE/RPE vs TUM-VI ground truth |
| **5 — Resolution sensitivity** | `notebooks/04_resolution_sensitivity.ipynb` | ATE/RPE/memory/time at 224–518 px input resolutions |
| **6 — Improvement strategies** | `notebooks/05_adaptive_resolution.ipynb` | Adaptive patching + progressive coarse-to-fine depth refinement |

---

## Repository structure

```
vggt-eval/
│
├── src/                          Core library
│   ├── pipeline.py               VGGTPipeline: model load, inference, PLY/COLMAP export
│   ├── chunking.py               SlidingWindowProcessor + Procrustes chunk alignment
│   ├── metrics.py                ATE, RPE, rotation error, AUC, MemoryProfiler, Timer
│   ├── visualization.py          Plotly point cloud, depth grids, cameras, trajectory, PLY I/O
│   ├── imu.py                    SO(3) math, gyro pre-integration, EuRoC CSV parsers
│   ├── tum_vi.py                 TUM-VI downloader (HTTP Range, ~35 MB selective fetch)
│   ├── imu_fusion.py             SLERP-based VGGT+IMU fusion, alpha_sweep helper
│   ├── resolution_sweep.py       ResolutionSweeper: sweep across input resolutions
│   └── improvements/
│       ├── adaptive.py           AdaptiveResolutionVGGT: global low-res + patch refinement
│       └── progressive.py        ProgressiveRefinement: coarse-to-fine depth pyramid
│
├── notebooks/
│   ├── 01_pipeline_basic.ipynb           Phase 2: end-to-end pipeline
│   ├── 02_sequential_processing.ipynb    Phase 3: sliding window
│   ├── 03_imu_integration.ipynb          Phase 4: IMU fusion
│   ├── 04_resolution_sensitivity.ipynb   Phase 5: resolution sweep
│   └── 05_adaptive_resolution.ipynb      Phase 6: improvement strategies
│
├── experiments/
│   └── configs/
│       ├── resolution_sweep.yaml         Hyperparameters for Phase 5
│       └── adaptive_resolution.yaml      Hyperparameters for Phase 6
│
└── results/                      Output CSVs, figures, PLY files (git-ignored)
```

---

## Quick start (Kaggle / Colab)

Each notebook is self-contained. Run the **Setup** cell first (Cell 0); it installs dependencies, clones VGGT and this repo, then restarts the kernel.  After the restart, run all remaining cells in order.

```python
# The setup cell does this automatically:
pip install --upgrade numpy
pip install plotly pandas pillow scipy matplotlib
git clone https://github.com/facebookresearch/vggt.git
git clone https://github.com/insha-py/vggt-eval.git
```

> **Note (Kaggle):** The kernel restart after `pip install --upgrade numpy` is mandatory to avoid a numpy ABI mismatch.  The setup cell handles this automatically via `IPython.Application.instance().kernel.do_shutdown(True)`.

---

## Phases in detail

### Phase 2 — Basic VGGT Pipeline

**Notebook:** `01_pipeline_basic.ipynb`

Runs VGGT on a folder of images and produces:
- `(N, 3, 4)` camera extrinsics (world-to-camera, OpenCV convention)
- `(N, H, W, 1)` metric depth maps + confidence
- Dense point cloud as `.ply`
- COLMAP-compatible sparse reconstruction

Also benchmarks **inference time** and **GPU memory** vs frame count.

**Key class:** `src/pipeline.py → VGGTPipeline`

---

### Phase 3 — Sliding Window Processing

**Notebook:** `02_sequential_processing.ipynb`

Addresses the O(N²) attention cost of VGGT for long sequences (>20 frames exceeds T4 VRAM).  Processes frames in overlapping windows and aligns consecutive chunks via **Procrustes / Umeyama similarity alignment** on the overlap region.

**Key class:** `src/chunking.py → SlidingWindowProcessor`

---

### Phase 4 — IMU Integration

**Notebook:** `03_imu_integration.ipynb`

**Dataset:** TUM Visual-Inertial Dataset — `room1` EuRoC 512-px export, 40 frames.  Download is ~35 MB via selective HTTP Range requests (not the full 1.6 GB archive).

**Fusion strategy:**
1. Extract relative rotations from VGGT: `R_rel_vggt[i] = R_v[i+1] @ R_v[i]ᵀ`
2. Integrate gyroscope between image frames (midpoint integration).
3. SLERP-blend: `R_rel_fused = slerp(R_rel_vggt, R_rel_imu_cam, α)`
   - `α=0` → pure VGGT,  `α=1` → pure IMU
4. Reconstruct absolute poses by chaining from the first VGGT pose.
5. Keep VGGT camera centres (translation unchanged).

An `alpha_sweep` over `α ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}` finds the optimal blend weight for ATE on the TUM-VI ground truth.

**Key classes:** `src/imu.py → IMUPreintegrator`, `src/imu_fusion.py → IMUVGGTFusion`

---

### Phase 5 — Resolution Sensitivity Analysis

**Notebook:** `04_resolution_sensitivity.ipynb`

**Core thesis contribution.**  VGGT was trained at 518 px square inputs; this phase quantifies the accuracy cost of using lower resolutions.

Runs VGGT at `[224, 280, 336, 392, 448, 518]` px and records:

| Metric | Description |
|--------|-------------|
| ATE (mean/RMSE/median) | Absolute Trajectory Error after Umeyama alignment |
| RPE translation | Relative Pose Error — translation |
| RPE rotation | Relative Pose Error — rotation (°) |
| Mean rotation error | Per-frame rotation vs GT |
| Inference time (s) | Wall-clock time |
| Peak GPU memory (MB) | Maximum VRAM usage |
| Mean depth confidence | Average VGGT depth confidence score |

**Key class:** `src/resolution_sweep.py → ResolutionSweeper`

---

### Phase 6 — Improvement Strategies

**Notebook:** `05_adaptive_resolution.ipynb`

Two strategies to recover 518-px quality at lower average cost:

#### Adaptive Resolution (`src/improvements/adaptive.py`)

1. **Global low-res pass** at 224 px — cheap, full scene.
2. **Identify uncertain regions**: find patches where mean depth confidence < threshold.
3. **High-res patch inference** at 518 px on those patches only.
4. **Confidence-weighted merge**: blend refined depth back into the global map.

Camera poses come from the global pass; only depth maps are refined.

#### Progressive Refinement (`src/improvements/progressive.py`)

Multi-scale pyramid `224 → 336 → 518 px`:
- At each finer level, upscale the accumulated depth and blend with the new prediction using confidence weights.
- Final poses are from the 518-px pass; depth is the weighted composite.

---

## Key design decisions

- **No accelerometer integration** — gravity alignment + bias estimation on a T4 with 40 frames is unreliable.  Gyro-only gives clean rotation priors without drift-accumulation risk.
- **HTTP Range download** — TUM-VI archives are 1.6 GB.  Five 16 MB scan windows find tar headers for the 43 needed files; total download ≈ 35 MB.
- **Integer timestamp parsing** — TUM-VI filenames are 19-digit nanosecond integers. Converting through `float64` loses 3–4 digits, breaking filename lookups.  All timestamp-to-filename mapping uses `int(p.stem)` directly.
- **Kernel restart on Kaggle** — upgrading numpy mid-session causes a C ABI mismatch.  The setup cell forces a kernel restart; imports and sys.path are re-established in Cell 1.

---

## Results (expected)

Results CSVs are written to `results/` (git-ignored).  Representative numbers on TUM-VI `room1` (40 frames, T4 GPU):

| Phase | Key result |
|-------|-----------|
| Phase 4 | Best α ≈ 0.1–0.3; IMU slightly improves rotation in fast-rotation windows |
| Phase 5 | ATE roughly doubles from 518 px → 224 px; inference time halves |
| Phase 6 | Adaptive ≈ 518-px ATE at ~60–70 % of 518-px time (scene-dependent) |

---

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0 + CUDA
- `numpy`, `scipy`, `pandas`, `pillow`, `matplotlib`, `plotly`
- `vggt` (Facebook Research, auto-cloned by setup cells)
- Optional: `open3d`, `trimesh`, `pycolmap` (for point cloud / COLMAP export)
