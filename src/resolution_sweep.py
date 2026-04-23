"""
Resolution Sensitivity Analysis (Phase 5 — core thesis contribution).

VGGT was trained with 518 px square inputs. This module measures how much
pose accuracy and reconstruction quality degrade when images are downscaled
before inference — useful for understanding the model's resolution bottleneck
and motivating the adaptive / progressive improvement strategies.

Sweep logic
-----------
For each resolution r in [224, 280, 336, 392, 448, 518]:
  1. Resize images to (r, r) with square-padding.
  2. Run VGGT inference.
  3. Record: ATE, RPE, rotation error, inference time, peak GPU memory.
  4. Optionally align predictions against ground-truth before computing errors.

The results DataFrame can be passed directly to the visualisation helpers in
src/visualization.py for plotting error-vs-resolution curves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default resolution ladder (pixels, square)
DEFAULT_RESOLUTIONS = [224, 280, 336, 392, 448, 518]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    resolution: int
    ate_mean:   float = float("nan")
    ate_rmse:   float = float("nan")
    ate_median: float = float("nan")
    rpe_trans:  float = float("nan")
    rpe_rot:    float = float("nan")
    rot_mean:   float = float("nan")     # mean rotation error (degrees)
    time_s:     float = float("nan")     # wall-clock inference time
    peak_mb:    float = float("nan")     # peak GPU memory
    n_frames:   int   = 0
    extrinsics: Optional[np.ndarray] = field(default=None, repr=False)
    depth_conf_mean: float = float("nan")  # mean depth confidence


# ---------------------------------------------------------------------------
# ResolutionSweeper
# ---------------------------------------------------------------------------

class ResolutionSweeper:
    """
    Run VGGT at multiple input resolutions and collect quality metrics.

    Parameters
    ----------
    resolutions : list of ints
        Input sizes (square) to evaluate.  Default: [224,280,336,392,448,518].
    conf_thresh : float
        Depth confidence threshold forwarded to the pipeline.
    store_extrinsics : bool
        If True, each ResolutionResult keeps the raw (N,3,4) extrinsics array
        so callers can do further analysis.  Increases memory usage.
    chunk_size : int or None
        If set, use sliding-window processing instead of a single VGGT pass.
        Recommended: 8 (safe at all resolutions on a T4 16 GB GPU).
        Allows running 40+ frames without OOM at the cost of Procrustes
        alignment error between chunks.
    overlap : int
        Overlap frames between adjacent chunks (only used when chunk_size
        is set).  Minimum 2; 3 gives good alignment.

    Example
    -------
    # Single-pass (20 frames, no OOM risk):
    sweeper = ResolutionSweeper()

    # Sliding-window (40 frames, OOM-safe at all resolutions):
    sweeper = ResolutionSweeper(chunk_size=8, overlap=3)

    sweeper.load_model()
    results = sweeper.run(image_dir="path/to/frames/", gt_extrinsics=gt)
    df = sweeper.to_dataframe(results)
    """

    def __init__(
        self,
        resolutions: List[int] = DEFAULT_RESOLUTIONS,
        conf_thresh: float = 5.0,
        store_extrinsics: bool = False,
        chunk_size: Optional[int] = None,
        overlap: int = 3,
    ):
        if chunk_size is not None and chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.resolutions      = resolutions
        self.conf_thresh      = conf_thresh
        self.store_extrinsics = store_extrinsics
        self.chunk_size       = chunk_size
        self.overlap          = overlap
        self._model           = None
        self._device          = None
        self._dtype           = None

    # ------------------------------------------------------------------
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load VGGT model (reuses pipeline helper)."""
        from src.pipeline import VGGTPipeline
        pipe = VGGTPipeline()
        pipe.load_model(checkpoint_path=checkpoint_path)
        self._model  = pipe.model
        self._device = pipe.device
        self._dtype  = pipe.dtype
        print(f"[ResolutionSweeper] Model loaded on {self._device}")

    # ------------------------------------------------------------------
    def run(
        self,
        image_dir: str,
        max_frames: Optional[int] = None,
        gt_extrinsics: Optional[np.ndarray] = None,
        align: bool = True,
        with_scale: bool = True,
    ) -> List[ResolutionResult]:
        """
        Sweep over all resolutions and return a list of ResolutionResult.

        Parameters
        ----------
        image_dir      : directory of input images
        max_frames     : subsample to at most this many frames
        gt_extrinsics  : (N, 3, 4) ground-truth poses for ATE/RPE;
                         if None, pose metrics will be NaN
        align          : Umeyama-align before computing ATE
        with_scale      : estimate scale during alignment
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        from src.pipeline       import load_images_from_dir, run_vggt_inference
        from src.metrics        import (
            compute_ate, compute_rpe, compute_rotation_errors,
            MemoryProfiler, Timer,
        )

        # Load original images once; resize per resolution below
        from PIL import Image as PILImage
        import glob, os, torch

        exts  = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        paths: list = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(image_dir, e)))
        paths = sorted(set(paths))
        if max_frames:
            paths = paths[:max_frames]
        if not paths:
            raise ValueError(f"No images found in {image_dir}")

        import gc
        import torch

        results: List[ResolutionResult] = []

        for res in self.resolutions:
            print(f"[ResolutionSweeper] resolution={res}px  ({len(paths)} frames)"
                  + (f"  [chunked: size={self.chunk_size} overlap={self.overlap}]"
                     if self.chunk_size else ""))

            with MemoryProfiler() as mem, Timer() as tmr:
                if self.chunk_size:
                    ext, conf_mean = self._run_chunked(paths, res)
                else:
                    imgs, _, _ = load_images_from_dir(
                        image_dir, target_size=res, max_frames=max_frames
                    )
                    out = run_vggt_inference(
                        self._model, imgs, self._device, self._dtype, resolution=res
                    )
                    ext       = out["extrinsic"]
                    conf_mean = float(np.mean(out["depth_conf"]))
                    del imgs, out
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            rr = ResolutionResult(
                resolution      = res,
                n_frames        = len(ext),
                time_s          = tmr.elapsed,
                peak_mb         = mem.peak_mb,
                depth_conf_mean = conf_mean,
            )

            if self.store_extrinsics:
                rr.extrinsics = ext

            if gt_extrinsics is not None:
                gt = gt_extrinsics[:len(ext)]

                ate = compute_ate(ext, gt, align=align, with_scale=with_scale)
                rpe = compute_rpe(ext, gt, step=1)
                rot = compute_rotation_errors(ext, gt)

                rr.ate_mean   = ate["mean"]
                rr.ate_rmse   = ate["rmse"]
                rr.ate_median = ate["median"]
                rr.rpe_trans  = rpe["trans_mean"]
                rr.rpe_rot    = rpe["rot_mean"]
                rr.rot_mean   = rot["mean"]

            results.append(rr)
            print(
                f"  time={rr.time_s:.1f}s  peak={rr.peak_mb:.0f}MB"
                + (f"  ATE={rr.ate_mean:.4f}" if gt_extrinsics is not None else "")
            )

        return results

    # ------------------------------------------------------------------
    def _run_chunked(
        self,
        paths: list,
        resolution: int,
    ) -> tuple:
        """
        Run VGGT in sliding-window chunks at `resolution` px and stitch
        poses with Procrustes alignment on the overlap frames.

        Returns (extrinsics (N,3,4), mean_depth_conf float).
        """
        import gc
        import torch
        from src.pipeline  import load_images_from_list, run_vggt_inference
        from src.chunking  import align_chunk_to_reference

        step   = self.chunk_size - self.overlap
        starts = list(range(0, len(paths), step))
        pairs  = [(s, min(s + self.chunk_size, len(paths))) for s in starts]
        pairs  = [(s, e) for s, e in pairs if (e - s) > self.overlap]

        frame_extrs: Dict[int, np.ndarray] = {}
        conf_acc = 0.0

        for ci, (s, e) in enumerate(pairs):
            chunk_paths = paths[s:e]
            imgs, _ = load_images_from_list(chunk_paths, target_size=resolution)

            out = run_vggt_inference(
                self._model, imgs, self._device, self._dtype, resolution=resolution
            )
            ext  = out["extrinsic"]        # (K, 3, 4)
            conf = float(np.mean(out["depth_conf"]))
            conf_acc += conf * len(chunk_paths)

            if ci == 0:
                for local_i, global_i in enumerate(range(s, e)):
                    frame_extrs[global_i] = ext[local_i]
            else:
                ref_ov = np.stack([frame_extrs[i] for i in range(s, s + self.overlap)])
                aligned, _, info = align_chunk_to_reference(
                    ref_ov, ext[:self.overlap],
                    ext, np.zeros((0, 3)),
                )
                print(f"    chunk {ci}: align residual={info['residual_m']:.4f} m")
                for local_i, global_i in enumerate(range(s, e)):
                    frame_extrs[global_i] = aligned[local_i]

            del imgs, out, ext
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        combined = np.stack([frame_extrs[i] for i in range(len(paths))])
        return combined, conf_acc / len(paths)

    # ------------------------------------------------------------------
    def run_from_tensors(
        self,
        images_per_res: Dict[int, "torch.Tensor"],
        gt_extrinsics: Optional[np.ndarray] = None,
        align: bool = True,
        with_scale: bool = True,
    ) -> List[ResolutionResult]:
        """
        Alternative entry point when callers pre-load images at each resolution.

        images_per_res : {resolution: (N, 3, res, res) float32 tensor}
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        from src.pipeline import run_vggt_inference
        from src.metrics  import (
            compute_ate, compute_rpe, compute_rotation_errors,
            MemoryProfiler, Timer,
        )

        results: List[ResolutionResult] = []
        for res, imgs in images_per_res.items():
            print(f"[ResolutionSweeper] resolution={res}px")
            with MemoryProfiler() as mem, Timer() as tmr:
                out = run_vggt_inference(
                    self._model, imgs, self._device, self._dtype, resolution=res
                )
            ext  = out["extrinsic"]
            conf = out["depth_conf"]

            rr = ResolutionResult(
                resolution      = res,
                n_frames        = len(ext),
                time_s          = tmr.elapsed,
                peak_mb         = mem.peak_mb,
                depth_conf_mean = float(np.mean(conf)),
            )
            if self.store_extrinsics:
                rr.extrinsics = ext

            if gt_extrinsics is not None:
                gt  = gt_extrinsics[:len(ext)]
                ate = compute_ate(ext, gt, align=align, with_scale=with_scale)
                rpe = compute_rpe(ext, gt, step=1)
                rot = compute_rotation_errors(ext, gt)
                rr.ate_mean   = ate["mean"]
                rr.ate_rmse   = ate["rmse"]
                rr.ate_median = ate["median"]
                rr.rpe_trans  = rpe["trans_mean"]
                rr.rpe_rot    = rpe["rot_mean"]
                rr.rot_mean   = rot["mean"]

            results.append(rr)
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

        return results

    # ------------------------------------------------------------------
    @staticmethod
    def to_dataframe(results: List[ResolutionResult]):
        """Convert list of ResolutionResult to a pandas DataFrame."""
        import pandas as pd
        rows = []
        for r in results:
            rows.append({
                "resolution":       r.resolution,
                "ate_mean":         r.ate_mean,
                "ate_rmse":         r.ate_rmse,
                "ate_median":       r.ate_median,
                "rpe_trans":        r.rpe_trans,
                "rpe_rot":          r.rpe_rot,
                "rot_mean_deg":     r.rot_mean,
                "time_s":           r.time_s,
                "peak_mb":          r.peak_mb,
                "n_frames":         r.n_frames,
                "depth_conf_mean":  r.depth_conf_mean,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def plot_sweep(results: List[ResolutionResult], save_dir: Optional[str] = None):
        """
        Quick matplotlib summary: ATE, RPE, time, memory vs resolution.
        Returns the Figure object.
        """
        import matplotlib.pyplot as plt

        resolutions = [r.resolution for r in results]
        has_gt = not np.isnan(results[0].ate_mean)

        n_rows = 2 if has_gt else 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows, 2)

        # Row 0: time + memory
        ax = axes[0, 0]
        ax.plot(resolutions, [r.time_s for r in results], "o-", color="steelblue")
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Inference time (s)")
        ax.set_title("Inference Time vs Resolution")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(resolutions, [r.peak_mb for r in results], "o-", color="darkorange")
        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("Peak GPU memory (MB)")
        ax.set_title("Peak GPU Memory vs Resolution")
        ax.grid(True, alpha=0.3)

        if has_gt:
            # Row 1: ATE + RPE
            ax = axes[1, 0]
            ax.plot(resolutions, [r.ate_mean   for r in results], "o-", label="ATE mean")
            ax.plot(resolutions, [r.ate_rmse   for r in results], "s--", label="ATE RMSE")
            ax.plot(resolutions, [r.ate_median for r in results], "^:", label="ATE median")
            ax.set_xlabel("Resolution (px)")
            ax.set_ylabel("ATE (m)")
            ax.set_title("ATE vs Resolution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(resolutions, [r.rpe_trans for r in results], "o-", label="RPE trans (m)")
            ax2 = ax.twinx()
            ax2.plot(resolutions, [r.rpe_rot for r in results], "s--r", label="RPE rot (°)")
            ax.set_xlabel("Resolution (px)")
            ax.set_ylabel("RPE translation (m)")
            ax2.set_ylabel("RPE rotation (°)")
            ax.set_title("RPE vs Resolution")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "resolution_sweep.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {path}")

        return fig
