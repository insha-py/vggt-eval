"""
Sequential / sliding-window frame processing for VGGT (Phase 3).

Problem
-------
VGGT's global self-attention is O(N²) in the number of frames, so processing
long sequences (100+ frames) quickly exceeds 16 GB VRAM.

Solution: sliding window
------------------------
Split the sequence into overlapping chunks, run VGGT per chunk, then
stitch the per-chunk reconstructions into a single coordinate frame using
rigid-body (Procrustes) alignment on the overlap cameras.

Usage
-----
from src.chunking import SlidingWindowProcessor

proc = SlidingWindowProcessor(pipeline, chunk_size=10, overlap=2)
result = proc.process(image_dir="path/to/images/")

# result keys: extrinsic, intrinsic, depth_map, depth_conf,
#              point_cloud, point_colors, chunk_stats
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# torch is required only at runtime (Colab/Kaggle); defer so pure-numpy helpers
# remain importable in environments without torch (e.g. local dev machines).
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data container for one processed chunk
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_idx:   int
    frame_start: int
    frame_end:   int            # exclusive
    extrinsic:   np.ndarray     # (K, 3, 4)
    intrinsic:   np.ndarray     # (K, 3, 3)
    depth_map:   np.ndarray     # (K, H, W, 1)
    depth_conf:  np.ndarray     # (K, H, W)
    point_cloud: np.ndarray     # (M, 3)
    point_colors: Optional[np.ndarray]  # (M, 3) uint8 or None
    inference_time_s: float
    peak_gpu_mb: float


# ---------------------------------------------------------------------------
# Rigid-body alignment helpers
# ---------------------------------------------------------------------------

def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """(N,3,4) -> (N,3) camera centres in world frame."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3,  3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


def estimate_rigid_transform_procrustes(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate rigid transform (R, t, s) such that dst ≈ s * R @ src + t,
    using SVD (Umeyama / Procrustes).

    Args:
        src_points: (N, 3)  source points  (chunk B's overlap camera centres)
        dst_points: (N, 3)  target points  (chunk A's overlap camera centres)

    Returns:
        R (3,3), t (3,), s (float scalar)
    """
    assert src_points.shape == dst_points.shape, \
        f"Shape mismatch: {src_points.shape} vs {dst_points.shape}"
    N = len(src_points)

    mu_s = src_points.mean(0)
    mu_d = dst_points.mean(0)
    src_c = src_points - mu_s
    dst_c = dst_points - mu_d

    var_s = (src_c ** 2).sum() / N
    H = (dst_c.T @ src_c) / N

    U, D, Vt = np.linalg.svd(H)
    # Ensure right-handed
    sign_diag = np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))])
    R = U @ sign_diag @ Vt
    s = (D * np.diag(sign_diag)).sum() / var_s if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def transform_extrinsics(
    extrinsics: np.ndarray,
    R_T: np.ndarray,
    t_T: np.ndarray,
    s_T: float = 1.0,
) -> np.ndarray:
    """
    Apply world-frame transform T = (R_T, t_T, s_T) to a batch of extrinsics.

    Each extrinsic E maps: world_B -> camera.
    We want:               world_A -> camera.
    T maps world_B -> world_A:  p_A = s_T * R_T @ p_B + t_T

    New extrinsic R_new = R_pred @ R_T^T / s_T ... simplification below.
    """
    aligned = extrinsics.copy().astype(np.float64)
    R_T_f64 = R_T.astype(np.float64)
    for i in range(len(aligned)):
        R_pred = aligned[i, :3, :3]
        t_pred = aligned[i, :3,  3]

        c_B = -R_pred.T @ t_pred                              # centre in world_B
        c_A = s_T * R_T_f64 @ c_B + t_T.astype(np.float64)  # centre in world_A

        R_new = R_pred @ R_T_f64.T / s_T   # absorb scale into rotation column norms
        # Renormalise rows (keep it a proper rotation for each column scale=1/s)
        # Actually for similarity we keep: R_new (not orthogonal in general)
        # To keep R as a proper rotation and encode scale in t only:
        # We DON'T scale R; scale only affects t:
        R_new = R_pred @ R_T_f64.T          # rotation part (orthogonal)
        t_new = -R_new @ c_A               # recompute t from aligned centre

        aligned[i, :3, :3] = R_new
        aligned[i, :3,  3] = t_new

    return aligned.astype(np.float32)


def transform_point_cloud(
    points: np.ndarray,
    R_T:    np.ndarray,
    t_T:    np.ndarray,
    s_T:    float = 1.0,
) -> np.ndarray:
    """Apply similarity transform p_A = s*R@p_B + t to (N,3) points."""
    return (s_T * (R_T @ points.T).T + t_T).astype(np.float32)


def align_chunk_to_reference(
    ref_extrinsics:    np.ndarray,  # overlap frames from chunk A (aligned frame)
    new_extrinsics:    np.ndarray,  # overlap frames from chunk B (its own frame)
    new_all_extrinsics: np.ndarray, # ALL frames of chunk B
    new_points:        np.ndarray,  # point cloud of chunk B
    estimate_scale:    bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Align chunk B into chunk A's world frame using Procrustes on overlap cameras.

    Returns:
        aligned_extrinsics (K_B, 3, 4)
        aligned_points     (M_B, 3)
        info dict          {R, t, s, residual_m}
    """
    src = _camera_centers(new_extrinsics)   # overlap centres in B's frame
    dst = _camera_centers(ref_extrinsics)   # overlap centres in A's frame

    if len(src) == 1:
        # Only 1 overlap frame: use direct formula (no scale)
        R_pred = new_extrinsics[0, :3, :3]
        t_pred = new_extrinsics[0, :3,  3]
        R_ref  = ref_extrinsics[0, :3, :3]
        t_ref  = ref_extrinsics[0, :3,  3]

        # T such that world_B -> world_A: E_ref = E_B @ inv(T)
        # R_T = R_A^T @ R_B
        # t_T = R_A^T @ (t_B - t_A)  ... using first frame
        R_T = R_ref.T @ R_pred
        t_T = R_ref.T @ (t_pred - t_ref)
        s_T = 1.0
    else:
        R_T, t_T, s_T = estimate_rigid_transform_procrustes(src, dst)
        if not estimate_scale:
            s_T = 1.0

    # Residual after alignment
    src_aligned = s_T * (R_T @ src.T).T + t_T
    residual = float(np.linalg.norm(src_aligned - dst, axis=1).mean())

    aligned_extrs  = transform_extrinsics(new_all_extrinsics, R_T, t_T, s_T)
    aligned_points = transform_point_cloud(new_points, R_T, t_T, s_T)

    return aligned_extrs, aligned_points, {"R": R_T, "t": t_T, "s": s_T,
                                           "residual_m": residual}


# ---------------------------------------------------------------------------
# Sliding-window processor
# ---------------------------------------------------------------------------

class SlidingWindowProcessor:
    """
    Process a long image sequence with VGGT using a sliding window.

    Parameters
    ----------
    pipeline   : VGGTPipeline instance (model already loaded)
    chunk_size : number of frames per chunk
    overlap    : number of frames shared between adjacent chunks
                 (used for alignment)
    conf_thresh: depth-confidence threshold for point cloud building
    estimate_scale: whether to estimate scale between chunks
                    (set False if you trust that VGGT produces consistent scale)
    """

    def __init__(
        self,
        pipeline,
        chunk_size:     int   = 10,
        overlap:        int   = 2,
        conf_thresh:    float = 5.0,
        estimate_scale: bool  = True,
    ):
        assert overlap >= 1, "Need at least 1 overlap frame for alignment."
        assert chunk_size > overlap, "chunk_size must be greater than overlap."

        self.pipeline       = pipeline
        self.chunk_size     = chunk_size
        self.overlap        = overlap
        self.conf_thresh    = conf_thresh
        self.estimate_scale = estimate_scale

    # ------------------------------------------------------------------
    def _frame_indices(self, total: int) -> List[Tuple[int, int]]:
        """
        Return list of (start, end) frame index pairs (end exclusive).
        """
        step   = self.chunk_size - self.overlap
        starts = list(range(0, total, step))
        pairs  = [(s, min(s + self.chunk_size, total)) for s in starts]
        # Drop chunks shorter than overlap+1 (can't align)
        pairs = [(s, e) for s, e in pairs if (e - s) > self.overlap]
        return pairs

    # ------------------------------------------------------------------
    def _run_chunk(
        self,
        image_paths: List[str],
        chunk_idx:   int,
        frame_start: int,
        frame_end:   int,
    ) -> ChunkResult:
        """Run VGGT on a single chunk of frames."""
        from src.pipeline import load_images_from_list, run_vggt_inference, depth_to_point_cloud

        paths = image_paths[frame_start:frame_end]
        images, _ = load_images_from_list(paths)

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0  = time.perf_counter()
        raw = run_vggt_inference(
            self.pipeline.model, images,
            self.pipeline.device, self.pipeline.dtype,
        )
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if (_TORCH_AVAILABLE and torch.cuda.is_available()) else 0.0)

        pts, clrs = depth_to_point_cloud(
            raw["depth_map"], raw["depth_conf"],
            raw["extrinsic"], raw["intrinsic"],
            images_rgb=images.numpy(),
            conf_thresh=self.conf_thresh,
        )

        return ChunkResult(
            chunk_idx=chunk_idx,
            frame_start=frame_start,
            frame_end=frame_end,
            extrinsic=raw["extrinsic"],
            intrinsic=raw["intrinsic"],
            depth_map=raw["depth_map"],
            depth_conf=raw["depth_conf"],
            point_cloud=pts,
            point_colors=clrs,
            inference_time_s=elapsed,
            peak_gpu_mb=peak_mb,
        )

    # ------------------------------------------------------------------
    def process(
        self,
        image_dir:  Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        verbose:    bool = True,
    ) -> Dict:
        """
        Process a long image sequence using sliding window.

        Supply either image_dir OR image_paths.

        Returns
        -------
        dict with:
            extrinsic       (N_total, 3, 4) aligned extrinsics
            intrinsic       (N_total, 3, 3) intrinsics
            depth_map       list of per-chunk depth maps
            depth_conf      list of per-chunk confidence maps
            point_cloud     (M_total, 3) merged point cloud
            point_colors    (M_total, 3) uint8 or None
            chunk_results   list[ChunkResult]
            chunk_stats     list of alignment info dicts
            image_paths     list[str]
        """
        from src.pipeline import _list_images

        if image_paths is None:
            if image_dir is None:
                raise ValueError("Provide image_dir or image_paths.")
            image_paths = _list_images(image_dir)
        if not image_paths:
            raise ValueError("No images found.")

        total  = len(image_paths)
        pairs  = self._frame_indices(total)
        n_chunks = len(pairs)

        if verbose:
            print(f"[Chunking] {total} frames → {n_chunks} chunks "
                  f"(size={self.chunk_size}, overlap={self.overlap})")

        chunk_results: List[ChunkResult] = []
        chunk_stats:   List[Dict]        = []

        # ---- process first chunk (defines world frame) ----
        s0, e0 = pairs[0]
        if verbose:
            print(f"  Chunk 0/{n_chunks-1}  frames [{s0}:{e0}]")
        cr0 = self._run_chunk(image_paths, 0, s0, e0)
        chunk_results.append(cr0)
        chunk_stats.append({"aligned": False, "residual_m": 0.0})

        # Accumulators (initialised with chunk 0)
        all_extrs  = [cr0.extrinsic]
        all_intrs  = [cr0.intrinsic]
        all_pts    = [cr0.point_cloud]
        all_clrs   = [cr0.point_colors]

        # World-frame extrinsics for ALL frames processed so far
        # (index 0..e0-1 are chunk 0, already in chunk 0's world frame)
        frame_extrs = {i: cr0.extrinsic[i - s0] for i in range(s0, e0)}

        # ---- process remaining chunks ----
        for ci, (s, e) in enumerate(pairs[1:], start=1):
            if verbose:
                print(f"  Chunk {ci}/{n_chunks-1}  frames [{s}:{e}]")

            cr = self._run_chunk(image_paths, ci, s, e)

            # ---- alignment ----
            # Overlap is [s .. prev_end), which in current chunk = indices 0..overlap-1
            # and in the global aligned frame = frame_extrs[s..s+overlap-1]
            ov = self.overlap
            ref_extrs_ov  = np.stack([frame_extrs[i] for i in range(s, s + ov)])
            new_extrs_ov  = cr.extrinsic[:ov]

            aligned_extrs, aligned_pts, info = align_chunk_to_reference(
                ref_extrs_ov, new_extrs_ov,
                cr.extrinsic, cr.point_cloud,
                estimate_scale=self.estimate_scale,
            )

            if verbose:
                print(f"    Alignment residual: {info['residual_m']:.4f} m  "
                      f"scale={info['s']:.4f}")

            # Store aligned extrinsics for this chunk's non-overlap frames
            for local_i, global_i in enumerate(range(s, e)):
                frame_extrs[global_i] = aligned_extrs[local_i]

            # Only keep NON-overlap frames from this chunk (to avoid duplication)
            non_ov_slice = slice(ov, None)
            all_extrs.append(aligned_extrs[non_ov_slice])
            all_intrs.append(cr.intrinsic[non_ov_slice])
            all_pts.append(aligned_pts)
            if cr.point_colors is not None and all_clrs[-1] is not None:
                all_clrs.append(cr.point_colors)
            else:
                all_clrs.append(None)

            chunk_results.append(cr)
            chunk_stats.append(info)

            # ---- cleanup GPU memory between chunks ----
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- merge ----
        merged_extrs  = np.concatenate(all_extrs,  axis=0)
        merged_intrs  = np.concatenate(all_intrs,  axis=0)
        merged_pts    = np.concatenate(all_pts,     axis=0)
        merged_clrs   = (np.concatenate([c for c in all_clrs if c is not None], axis=0)
                         if any(c is not None for c in all_clrs) else None)

        if verbose:
            total_time = sum(cr.inference_time_s for cr in chunk_results)
            peak_max   = max(cr.peak_gpu_mb for cr in chunk_results)
            print(f"\n[Chunking] Done — {len(merged_pts):,} points  "
                  f"total inference time {total_time:.1f}s  "
                  f"max chunk peak {peak_max:.0f} MB")

        return dict(
            extrinsic    = merged_extrs,
            intrinsic    = merged_intrs,
            depth_map    = [cr.depth_map   for cr in chunk_results],
            depth_conf   = [cr.depth_conf  for cr in chunk_results],
            point_cloud  = merged_pts,
            point_colors = merged_clrs,
            chunk_results= chunk_results,
            chunk_stats  = chunk_stats,
            image_paths  = image_paths,
        )


# ---------------------------------------------------------------------------
# Parameter-sweep experiment helpers
# ---------------------------------------------------------------------------

def experiment_chunk_sizes(
    pipeline,
    image_paths: List[str],
    chunk_sizes: List[int] = (5, 10, 15, 20),
    overlap:     int = 2,
    conf_thresh: float = 5.0,
    max_frames:  int = 60,
) -> List[Dict]:
    """
    Run the sliding-window pipeline for each chunk_size and collect statistics.

    Returns list of result dicts, each augmented with summary fields.
    """
    paths = image_paths[:max_frames]
    records = []
    for cs in chunk_sizes:
        if cs <= overlap:
            continue
        print(f"\n{'='*50}")
        print(f"  chunk_size={cs}  overlap={overlap}  frames={len(paths)}")
        print(f"{'='*50}")
        proc = SlidingWindowProcessor(
            pipeline, chunk_size=cs, overlap=overlap, conf_thresh=conf_thresh
        )
        t0  = time.perf_counter()
        res = proc.process(image_paths=paths)
        total_time = time.perf_counter() - t0

        stats = {
            "chunk_size":      cs,
            "overlap":         overlap,
            "n_frames":        len(paths),
            "n_chunks":        len(res["chunk_results"]),
            "n_points":        len(res["point_cloud"]),
            "total_time_s":    total_time,
            "peak_gpu_mb":     max(cr.peak_gpu_mb for cr in res["chunk_results"]),
            "mean_residual_m": np.mean([s.get("residual_m", 0)
                                        for s in res["chunk_stats"]]),
        }
        print(f"  → {stats}")
        records.append({**stats, "result": res})
    return records


def experiment_overlaps(
    pipeline,
    image_paths: List[str],
    chunk_size:  int = 10,
    overlaps:    List[int] = (1, 2, 3),
    conf_thresh: float = 5.0,
    max_frames:  int = 60,
) -> List[Dict]:
    """
    Run the sliding-window pipeline for each overlap size.
    """
    paths = image_paths[:max_frames]
    records = []
    for ov in overlaps:
        if chunk_size <= ov:
            continue
        print(f"\n{'='*50}")
        print(f"  chunk_size={chunk_size}  overlap={ov}  frames={len(paths)}")
        print(f"{'='*50}")
        proc = SlidingWindowProcessor(
            pipeline, chunk_size=chunk_size, overlap=ov, conf_thresh=conf_thresh
        )
        t0  = time.perf_counter()
        res = proc.process(image_paths=paths)
        total_time = time.perf_counter() - t0

        stats = {
            "chunk_size":      chunk_size,
            "overlap":         ov,
            "n_frames":        len(paths),
            "n_chunks":        len(res["chunk_results"]),
            "n_points":        len(res["point_cloud"]),
            "total_time_s":    total_time,
            "peak_gpu_mb":     max(cr.peak_gpu_mb for cr in res["chunk_results"]),
            "mean_residual_m": np.mean([s.get("residual_m", 0)
                                        for s in res["chunk_stats"]]),
        }
        print(f"  → {stats}")
        records.append({**stats, "result": res})
    return records


# ---------------------------------------------------------------------------
# Memory profiling without running inference
# ---------------------------------------------------------------------------

def profile_memory_theoretical(
    frame_counts: List[int],
    base_mb:      float = 1880.0,
    coeff_mb:     float = 20.0,
) -> List[Dict]:
    """
    Estimate GPU memory from empirical formula:
        mem(N) ≈ base + coeff * N^1.5   (rough fit to VGGT paper Table 9)

    Falls back to published numbers where available.
    """
    # Values from VGGT paper Table 9
    known = {1: 1880, 10: 3630, 50: 11410, 200: 40630}

    results = []
    for N in frame_counts:
        if N in known:
            mb = float(known[N])
        else:
            # Interpolate / extrapolate with N^1.5 fit
            mb = base_mb + coeff_mb * (N ** 1.5)
        results.append({"n_frames": N, "estimated_mb": mb, "estimated_gb": mb / 1024})
    return results
