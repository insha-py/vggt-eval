# VGGT Evaluation 

**Evaluating VGGT (Visual Geometry Grounded Transformer) for real-world deployment:**
resolution sensitivity, metric scale recovery, and IMU-guided compute reduction.

> **Author:** Insha Khan  
> **Hardware:** Kaggle / Colab free-tier T4 GPU (14.6 GB VRAM)  
> **Dataset:** TUM Visual-Inertial Dataset (EuRoC format, 4 sequences)

---

## Motivation

### What VGGT is

VGGT (Wang et al., CVPR 2025 **Best Paper**) is a 1.2B-parameter feed-forward transformer that reconstructs camera poses, depth maps, point maps, and 3D point tracks from an unordered set of images in a single forward pass — no bundle adjustment, no feature matching, no iterative refinement. On standard benchmarks it runs in under 0.2 seconds for 10 views on an H100 and outperforms classical SfM pipelines on Co3D, RealEstate10K, DTU, and ETH3D.

Architecturally it uses a frozen **DINOv2 ViT-L/14** backbone (patch size 14 × 14) followed by 24 layers of alternating *within-frame* and *cross-frame* self-attention. Images are resized to a maximum of **518 px** — the native resolution of DINOv2 — before tokenisation. All four output heads (cameras, depth, point maps, tracks) are trained jointly; the paper shows this multi-task co-training substantially outperforms any single-task variant.

### Gaps the paper leaves open

Despite its impressive results, the VGGT paper leaves several practical deployment questions unanswered. These gaps directly motivated this thesis:

**1. Resolution is fixed at 518 px — what happens below?**  
The paper trains and evaluates exclusively at 518 px. It benchmarks performance on server-grade H100 GPUs. It never ablates sub-518 px inputs, never discusses what fraction of accuracy is preserved at lower resolutions, and never reports inference cost on consumer or free-tier GPU hardware. For practitioners who cannot afford an H100 — or for edge/mobile deployment where VRAM is a hard constraint — this is the central question.

**2. The evaluation metric hides metric scale.**  
VGGT reports pose accuracy as *AUC* of the minimum between Rotation Accuracy (RRA) and Translation Accuracy (RTA) at varying angular thresholds. This metric is **scale-invariant by construction** — it measures whether directions are correct, not whether magnitudes are correct. The paper does not report the Umeyama scale factor needed to convert VGGT's scene-normalised output to metric units, nor how stable that factor is across scenes or resolutions. Any downstream application that needs metric-scale poses (navigation, AR, robotics) must solve this problem independently.

**3. Compute cost ignores the free IMU.**  
VGGT is purely vision-based. The paper acknowledges that performance degrades under extreme camera rotations and that there is no native multi-sensor fusion. Yet every device that produces video in the wild — phones, robots, drones, AR headsets — also carries a low-cost IMU running at 200 Hz. The paper never asks: can the IMU, already present at zero marginal cost, be used to reduce the number of frames that need to be fed to VGGT?

**4. Frame budget and selection are not studied.**  
VGGT's compute scales quadratically with frame count (O(N²) attention). The paper benchmarks 1–200 frames on H100. At 8 frames on a T4 (14.6 GB), 518 px already consumes 9.5 GB. The paper is silent on strategies for selecting which frames to use when operating under a tight frame budget.

**5. Generalisation to VI datasets.**  
All standard benchmarks (Co3D, RealEstate10K, DTU) use internet-scraped or controlled-capture imagery. There is no evaluation on **visual-inertial datasets** where ground-truth trajectory comes from motion-capture and the IMU stream is co-recorded. This leaves open whether VGGT's accuracy transfers to the VI-SLAM use case.

---

## Dataset

**TUM Visual-Inertial (TUM-VI)** — EuRoC-format sequences recorded at 512 px with a synchronized IMU (200 Hz, BMI160 gyroscope + accelerometer) and motion-capture ground truth (~120 Hz).

| Sequence | Scene type | Motion profile | Duration |
|---|---|---|---|
| `room1` | Small indoor room | Slow handheld pan | ~10 s |
| `room2` | Small indoor room | Similar to room1 | ~10 s |
| `corridor1` | Long corridor | Fast, rotation-rich | ~15 s |
| `slides1` | Near-planar lecture screen | Fast horizontal pan | ~10 s |

All experiments use **8 frames** sampled from a 24-frame pool, evaluated against motion-capture ground truth using Umeyama-aligned ATE (Absolute Trajectory Error on camera centres, in metres).

Download is selective (~35 MB via HTTP Range requests) — not the full 1.6 GB archive.

---

## Experiments and Findings

### Phase 5 — Resolution Sensitivity

**Question:** Does VGGT's pose accuracy degrade when the input resolution is reduced below the training resolution of 518 px?

**Method:** Swept resolutions `[224, 280, 336, 392, 448, 518]` px. Ran VGGT at each resolution on all 4 sequences with 8 frames each. Measured ATE (Umeyama-aligned, with scale), inference time, and peak VRAM.

**Findings:**

| Finding | Detail |
|---|---|
| **ATE is resolution-invariant** | Identical ATE to 4 decimal places across 224–518 px for all 4 sequences. This is not an artefact of Umeyama alignment — it holds with and without scale correction. |
| **224 px saves ~3× time** | room1: 4.0 s at 518 px → ~0.6 s at 224 px (warm-model measurement, Phase 8) |
| **224 px saves 27% VRAM** | 9520 MB → 6983 MB; constant across all scenes (VRAM is determined by resolution × frames, not scene content) |
| **Savings scale with patch count** | VGGT tokenises at 14 × 14 patches; fewer patches = fewer attention queries = quadratically lower compute |

**Relation to paper:** The paper never ablates sub-518 px inputs. Our result shows VGGT's DINOv2 backbone retains sufficient scene representation at 224 px for the pose estimation head. The training resolution appears to be an upper bound on quality, not a requirement.

---

### Phase 7 — Metric Scale Analysis

**Question:** What Umeyama scale factor is needed to convert VGGT poses to metric units, and how does it vary across scenes and resolutions?

**Method:** For each sequence × resolution combination, extracted the Umeyama scale factor `s` (the ratio that maps VGGT's scene-normalised units to motion-capture metres).

**Findings:**

| Sequence | Mean scale `s` | CV (%) | Interpretation |
|---|---|---|---|
| `corridor1` | 1.86 | 38% | Close to metric by coincidence |
| `room1` | 19.2 | 38% | VGGT units ≈ 1/19 of metric |
| `room2` | 20.1 | 38% | Similar to room1 |
| `slides1` | 66.3 | 38% | Severe under-scaling |

Key results:

- **Scale varies 35× across scenes** (`corridor1` ≈ 1.9 vs `slides1` ≈ 66). There is no single conversion factor.
- **Scale is unstable within a scene across resolutions** — 38% coefficient of variation with a consistent non-monotonic pattern (dip at 280 px, peak at 448 px) identical across all 4 scenes. A single per-scene calibration measurement is insufficient; you would need measurements at each resolution.
- **ATE is unaffected** — the Umeyama alignment removes the scale before computing ATE, so trajectory shape is preserved even when absolute scale is wrong.

**Relation to paper:** The paper reports only AUC metrics (scale-invariant). Our work quantifies the practical scale problem for the first time on TUM-VI data. Any system integrating VGGT into a metric pipeline must account for this.

---

### Phase 8 — IMU-Guided Frame Selection

**Question:** Can the IMU gyroscope — available at zero marginal cost on VI hardware — be used to select more informative frames for VGGT, reducing compute without sacrificing accuracy?

**Motivation:** VGGT's permutation-equivariant architecture (for frames 2–N) means it is agnostic to which specific frames are chosen — only the geometric diversity of the selected set matters. Uniform time-based sampling wastes the frame budget on near-duplicate views during slow or static motion. The gyroscope measures cumulative rotation continuously at 200 Hz; we use it to gate frame selection: accept a new frame only when the camera has rotated ≥ θ° since the last accepted frame.

**Method:**

```
IMU cumulative rotation integration (SO(3) midpoint, 200 Hz)
  ↓
Accept frame i if: |rotation since last accepted frame| ≥ θ°
  ↓
Feed selected frames to VGGT (capped at MAX_FRAMES=8)
  ↓
Compare ATE vs uniform selection at same frame count
```

Swept θ ∈ {5°, 8°, 12°, 18°}. Compared three configurations: `uniform-518px` (baseline), `imu-518px` (selection only), `imu-224px` (selection + resolution).

**Room1 results (Phase 8, single-sequence):**

| Config | θ | ATE | ΔATE | Speedup | VRAM saved |
|---|---|---|---|---|---|
| Uniform 518px | — | 0.757 m | — | 1× | 0% |
| IMU-guided 518px | 5° | 0.659 m | −0.098 | 1.1× | 0% |
| IMU-guided 224px | 5° | **0.659 m** | **−0.098** | **6.2×** | **27%** |
| IMU-guided 224px | 12° | 0.772 m | +0.014 | 6.6× | 27% |

**Cross-sequence results (Phase 8b, all 4 scenes, θ=5°):**

| Sequence | Uniform-518px ATE | IMU-224px ATE | ΔATE | Speedup |
|---|---|---|---|---|
| `room1` | 0.757 m | 0.659 m | **−13%** | 7.0× |
| `room2` | 0.969 m | 0.747 m | **−23%** | 6.3× |
| `corridor1` | 0.833 m | 0.689 m | **−17%** | 6.4× |
| `slides1` | 1.190 m | 1.108 m | **−7%** | 6.5× |

**Findings:**

| Finding | Detail |
|---|---|
| **IMU selection improves ATE on all 4 scenes at θ=5°** | Gyro-gated selection picks rotation-rich frames; VGGT's triangulation geometry improves when views have diverse baseline angles |
| **Speedup is scene-independent** | 6.1–7.0× consistent across all 4 scene types — it is a mechanical consequence of 224px × 8 frames, not scene-dependent |
| **Resolution and accuracy are decoupled** | `imu-518px` and `imu-224px` have identical ATE at every point — confirming Phase 5's resolution-invariance extends to IMU-guided selection |
| **slides1 degrades above θ=8°** | Fast horizontal pan over a near-planar surface: at higher θ, selected frames are too spread out for VGGT to reconstruct planar content. θ=5° remains safe |
| **corridor1 is θ-invariant** | Fast continuous rotation fills the 8-frame budget identically for all θ values tested |
| **θ=5° is the robust operating point** | Beats uniform baseline on all 4 scenes; safest choice across diverse scene types |

**Relation to paper:** The VGGT paper does not mention IMU or frame selection. This is a novel contribution that exploits a sensor already present in VI hardware to simultaneously improve accuracy (better frame diversity) and reduce compute (lower resolution is safe).

---

## Key Takeaways

1. **Deploy at 224 px.** VGGT's accuracy is identical at 224–518 px. There is no empirical justification for running at native resolution for pose estimation. Running at 224 px saves 27% VRAM and reduces inference time by ~6× (warm model).

2. **Metric scale is not recovered.** The Umeyama scale factor varies 35× across scene types and is unstable within a scene (38% CV across resolutions). VGGT cannot be used as a drop-in metric pose estimator without a calibration step.

3. **IMU-guided selection at θ=5° is strictly better than uniform.** It improves ATE on all tested scene types while delivering 6–7× compute reduction. The only scenario where higher θ hurts is near-planar fast-pan content (slides1 at θ≥8°).

4. **VRAM savings are constant.** They depend only on resolution and frame count — not scene content. This makes the savings predictable and reliable for system design.

5. **The rotation convention in VGGT output is ambiguous.** We observe a persistent ~93.5° rotation error across all conditions after applying all standard GT corrections. This does not affect ATE (which uses only camera centres) but means rotation-dependent metrics from this evaluation should not be reported without further investigation.

---

## Results Summary Table

| Experiment | Metric | Result | vs. paper |
|---|---|---|---|
| Phase 5: Resolution sweep | ATE at 224–518 px | **Flat (invariant)** | Not studied in paper |
| Phase 5: Compute at 224 px | Time / VRAM | **~6× faster, 27% less VRAM** | Paper benchmarks H100 only |
| Phase 7: Umeyama scale | Scale factor `s` per sequence | **1.9–66×, 38% CV** | Paper uses scale-invariant AUC |
| Phase 8: IMU selection (room1) | ΔATE at θ=5°, imu-224px | **−13%, 6.2× speedup** | Not in paper |
| Phase 8b: IMU selection (4 scenes) | ΔATE at θ=5° | **−7% to −23%** on all scenes | Not in paper |

---

## Repository Structure

```
vggt-eval/
├── src/
│   ├── pipeline.py           VGGTPipeline: model load, inference, export
│   ├── metrics.py            ATE, RPE, rotation error, Umeyama alignment
│   ├── imu.py                SO(3) math, gyro integration, select_frames_by_rotation()
│   ├── tum_vi.py             TUM-VI HTTP Range downloader (~35 MB selective fetch)
│   ├── imu_fusion.py         SLERP-based VGGT+IMU pose blending
│   ├── chunking.py           SlidingWindowProcessor + Procrustes chunk alignment
│   ├── resolution_sweep.py   ResolutionSweeper across input sizes
│   └── visualization.py      Plotly trajectories, point clouds, depth grids
│
├── notebooks/
│   ├── 00_results_dashboard.ipynb     All findings in one place (run locally)
│   ├── 01_pipeline_basic.ipynb        End-to-end VGGT inference baseline
│   ├── 02_sequential_processing.ipynb Sliding window for long sequences
│   ├── 03_imu_integration.ipynb       SLERP fusion alpha sweep
│   ├── 04b_resolution_single.ipynb    Phase 5: single-pass resolution sweep
│   ├── 06_multisequence.ipynb         Phase 5: multi-sequence replication
│   ├── 07_metric_scale.ipynb          Phase 7: Umeyama scale analysis
│   └── 08_imu_frame_selection.ipynb   Phase 8: IMU-guided frame selection
│
└── results/
    ├── phase7_scale_across_sequences.csv
    ├── phase8_imu_frame_selection.csv
    └── phase8_multisequence.csv
```

---

## Running on Kaggle / Colab

Each notebook is self-contained. Run the **Setup** cell (Cell 0) first — it installs dependencies, clones VGGT and this repo, then restarts the kernel.

```python
# Setup cell does this automatically:
pip install --upgrade numpy
pip install plotly pandas pillow scipy matplotlib
git clone https://github.com/facebookresearch/vggt.git && pip install -e vggt
git clone https://github.com/insha-py/vggt-eval.git
```

For the dashboard notebook (`00_results_dashboard.ipynb`), run locally — it requires no inference, only `plotly` and `pandas`, and reads from `results/`.

---

## Reference

```bibtex
@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and
               Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {CVPR},
  year      = {2025}
}
```

Paper: [arXiv 2503.11651](https://arxiv.org/abs/2503.11651) · Project: [vgg-t.github.io](https://vgg-t.github.io)
