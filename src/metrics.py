"""
Evaluation metrics for VGGT 3D reconstruction.

Covers:
- Camera pose: ATE (Absolute Trajectory Error), RPE (Relative Pose Error),
  rotation error, AUC@k°
- 3D reconstruction: Chamfer distance, accuracy, completeness
- Runtime: GPU memory profiling, per-chunk timing
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pose alignment helpers
# ---------------------------------------------------------------------------

def align_trajectories_umeyama(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    with_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Umeyama similarity alignment: find (R, t, s) minimising
    ||s * R @ pred + t - gt||^2.

    Args:
        pred_positions: (N, 3) predicted camera centres
        gt_positions:   (N, 3) ground-truth camera centres
        with_scale:     if True estimate scale, otherwise fix s=1

    Returns:
        R (3,3), t (3,), s (scalar)
    """
    assert pred_positions.shape == gt_positions.shape
    N = pred_positions.shape[0]

    mu_p = pred_positions.mean(0)
    mu_g = gt_positions.mean(0)

    pred_c = pred_positions - mu_p
    gt_c   = gt_positions   - mu_g

    sigma_p = (pred_c ** 2).sum() / N
    H = (gt_c.T @ pred_c) / N

    U, D, Vt = np.linalg.svd(H)
    # Ensure right-handed coordinate system
    sign_diag = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = U @ sign_diag @ Vt

    s = (D * np.diag(sign_diag)).sum() / sigma_p if with_scale else 1.0
    t = mu_g - s * R @ mu_p
    return R, t, s


def apply_pose_alignment(
    pred_extrinsics: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
    s_align: float = 1.0,
) -> np.ndarray:
    """
    Apply a similarity transform to a batch of camera extrinsics [R|t] (N,3,4).

    The alignment is on camera centres: c_aligned = s * R_align @ c + t_align.
    The corresponding extrinsic becomes: R_new = R_pred @ R_align^T,
    t_new = -R_new @ c_aligned.
    """
    aligned = pred_extrinsics.copy()
    for i in range(len(pred_extrinsics)):
        R_pred = pred_extrinsics[i, :3, :3]
        t_pred = pred_extrinsics[i, :3, 3]
        c_pred = -R_pred.T @ t_pred

        c_aligned = s_align * R_align @ c_pred + t_align
        R_new     = R_pred @ R_align.T
        t_new     = -R_new @ c_aligned
        aligned[i, :3, :3] = R_new
        aligned[i, :3,  3] = t_new
    return aligned


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """Extract camera centres from extrinsics (N, 3, 4) -> (N, 3)."""
    R = extrinsics[:, :3, :3]  # (N,3,3)
    t = extrinsics[:, :3,  3]  # (N,3)
    # c = -R^T t  ->  einsum over batch
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


# ---------------------------------------------------------------------------
# ATE  — Absolute Trajectory Error
# ---------------------------------------------------------------------------

def compute_ate(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   np.ndarray,
    align: bool = True,
    with_scale: bool = False,
) -> Dict[str, float]:
    """
    Absolute Trajectory Error on camera centres.

    Args:
        pred_extrinsics: (N, 3, 4) predicted extrinsics
        gt_extrinsics:   (N, 3, 4) ground-truth extrinsics
        align:           apply Umeyama alignment before computing error
        with_scale:      estimate scale during alignment

    Returns:
        dict with keys: mean, rmse, median, std, max, scale
        scale: the Umeyama scale factor s (1.0 when align=False or with_scale=False).
               VGGT outputs normalised-scene-unit scale, not metric; s converts
               VGGT units → GT metric units.  s ≈ 1 means VGGT scale matches GT.
    """
    pred_c = _camera_centers(pred_extrinsics)
    gt_c   = _camera_centers(gt_extrinsics)

    s = 1.0
    if align:
        R, t, s = align_trajectories_umeyama(pred_c, gt_c, with_scale)
        pred_c  = s * (R @ pred_c.T).T + t

    errors = np.linalg.norm(pred_c - gt_c, axis=1)
    return {
        "mean":   float(np.mean(errors)),
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "median": float(np.median(errors)),
        "std":    float(np.std(errors)),
        "max":    float(np.max(errors)),
        "scale":  float(s),
    }


# ---------------------------------------------------------------------------
# RPE  — Relative Pose Error
# ---------------------------------------------------------------------------

def _relative_extrinsic(E_i: np.ndarray, E_j: np.ndarray) -> np.ndarray:
    """Relative extrinsic going from frame i to frame j (both 3x4)."""
    # Build 4x4 homogeneous matrices
    def to_4x4(E):
        M = np.eye(4)
        M[:3] = E
        return M
    return to_4x4(E_j) @ np.linalg.inv(to_4x4(E_i))


def compute_rpe(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   np.ndarray,
    step: int = 1,
) -> Dict[str, float]:
    """
    Relative Pose Error on consecutive frames separated by `step`.

    Returns:
        dict: trans_mean/rmse/median, rot_mean/rmse/median (rotation in degrees)
    """
    N = len(pred_extrinsics)
    trans_errors = []
    rot_errors   = []

    for i in range(0, N - step):
        j = i + step
        rel_pred = _relative_extrinsic(pred_extrinsics[i], pred_extrinsics[j])
        rel_gt   = _relative_extrinsic(gt_extrinsics[i],   gt_extrinsics[j])

        # Translation error (norm of translation difference)
        t_err = np.linalg.norm(rel_pred[:3, 3] - rel_gt[:3, 3])
        trans_errors.append(t_err)

        # Rotation error
        R_err = rel_pred[:3, :3] @ rel_gt[:3, :3].T
        rot_errors.append(_rotation_angle_deg(R_err))

    trans_errors = np.array(trans_errors)
    rot_errors   = np.array(rot_errors)

    return {
        "trans_mean":   float(np.mean(trans_errors)),
        "trans_rmse":   float(np.sqrt(np.mean(trans_errors ** 2))),
        "trans_median": float(np.median(trans_errors)),
        "rot_mean":     float(np.mean(rot_errors)),
        "rot_rmse":     float(np.sqrt(np.mean(rot_errors ** 2))),
        "rot_median":   float(np.median(rot_errors)),
    }


# ---------------------------------------------------------------------------
# Rotation error
# ---------------------------------------------------------------------------

def _rotation_angle_deg(R: np.ndarray) -> float:
    """Angle (degrees) for rotation matrix R (or relative rotation)."""
    # Clamp trace to [-1, 3] to avoid numerical issues with arccos
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_rotation_errors(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   np.ndarray,
) -> Dict[str, float]:
    """Per-frame rotation error between predicted and GT extrinsics."""
    errors = []
    for pred, gt in zip(pred_extrinsics, gt_extrinsics):
        R_err = pred[:3, :3] @ gt[:3, :3].T
        errors.append(_rotation_angle_deg(R_err))

    errors = np.array(errors)
    return {
        "mean":   float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std":    float(np.std(errors)),
        "max":    float(np.max(errors)),
    }


# ---------------------------------------------------------------------------
# AUC @ multiple thresholds — VGGT paper protocol (pairwise relative poses)
# ---------------------------------------------------------------------------

def compute_relative_pose_auc(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   np.ndarray,
    thresholds_deg:  List[float] = (5, 10, 20, 30),
) -> Dict[str, float]:
    """
    Pairwise Relative Pose AUC following the VGGT paper protocol.

    For every ordered pair (i, j) with i < j:
      - Relative Rotation Accuracy (RRA): geodesic angle between relative
        rotation matrices R_rel_pred and R_rel_gt.
      - Relative Translation Accuracy (RTA): angular error between the
        relative translation direction vectors t_rel_pred and t_rel_gt.
      - Per-pair error = max(RRA, RTA)  (both must be under threshold).

    AUC@θ = area under the accuracy-vs-threshold curve up to θ degrees,
    normalised to [0, 1].  Matches the AUC@30 reported in the VGGT paper.

    Returns:
        dict: auc@Xdeg for each threshold, and per-threshold acc@Xdeg.
    """
    N = len(pred_extrinsics)
    pair_errors: List[float] = []

    for i in range(N):
        for j in range(i + 1, N):
            # Relative rotation error
            R_rel_pred = pred_extrinsics[j, :3, :3] @ pred_extrinsics[i, :3, :3].T
            R_rel_gt   = gt_extrinsics[j,   :3, :3] @ gt_extrinsics[i,   :3, :3].T
            R_err      = R_rel_pred @ R_rel_gt.T
            rot_err    = _rotation_angle_deg(R_err)

            # Relative translation direction error (angular, scale-invariant)
            t_pred = pred_extrinsics[j, :3, 3] - pred_extrinsics[i, :3, 3]
            t_gt   = gt_extrinsics[j,   :3, 3] - gt_extrinsics[i,   :3, 3]
            n_pred = np.linalg.norm(t_pred)
            n_gt   = np.linalg.norm(t_gt)
            if n_pred < 1e-8 or n_gt < 1e-8:
                trans_err = 180.0
            else:
                cos_a = float(np.clip(np.dot(t_pred / n_pred, t_gt / n_gt), -1.0, 1.0))
                trans_err = float(np.degrees(np.arccos(cos_a)))

            pair_errors.append(max(rot_err, trans_err))

    pair_errors_arr = np.array(pair_errors)
    result: Dict[str, float] = {}
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    # Per-threshold accuracy
    thresholds_arr = np.array(thresholds_deg, dtype=float)
    fracs = []
    for thresh in thresholds_deg:
        frac = float(np.mean(pair_errors_arr < thresh))
        result[f"acc@{int(thresh)}deg"] = frac
        fracs.append(frac)

    # AUC normalised per threshold (one value per threshold ceiling)
    for thresh, frac_list in zip(thresholds_deg,
                                  [fracs[:k+1] for k in range(len(fracs))]):
        xs = thresholds_arr[:len(frac_list)]
        result[f"auc@{int(thresh)}deg"] = float(_trapz(frac_list, xs) / thresh)

    return result


def compute_auc(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   np.ndarray,
    thresholds_deg:  List[float] = (1, 2, 3, 5, 10, 15, 30),
    align: bool = True,
) -> Dict[str, float]:
    """
    AUC of per-frame absolute rotation errors (legacy helper).
    Use compute_relative_pose_auc for the paper-faithful pairwise metric.
    """
    rot_errors = []
    for pred, gt in zip(pred_extrinsics, gt_extrinsics):
        R_err = pred[:3, :3] @ gt[:3, :3].T
        rot_errors.append(_rotation_angle_deg(R_err))

    rot_errors = np.array(rot_errors)
    result: Dict[str, float] = {}

    fracs = []
    for thresh in thresholds_deg:
        frac = float(np.mean(rot_errors < thresh))
        result[f"acc@{thresh}deg"] = frac
        fracs.append(frac)

    thresholds_arr = np.array(thresholds_deg, dtype=float)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    result["auc"] = float(_trapz(fracs, thresholds_arr) / thresholds_arr[-1])
    return result


# ---------------------------------------------------------------------------
# 3-D reconstruction quality
# ---------------------------------------------------------------------------

def compute_chamfer_distance(
    pred_points: np.ndarray,
    gt_points:   np.ndarray,
    subsample:   Optional[int] = 50_000,
) -> Dict[str, float]:
    """
    Symmetric Chamfer distance between two point clouds.

    Args:
        pred_points: (M, 3)
        gt_points:   (N, 3)
        subsample:   if set, randomly subsample both clouds for speed

    Returns:
        dict: chamfer (symmetric average), accuracy (pred->gt), completeness (gt->pred)
    """
    if subsample and len(pred_points) > subsample:
        idx = np.random.choice(len(pred_points), subsample, replace=False)
        pred_points = pred_points[idx]
    if subsample and len(gt_points) > subsample:
        idx = np.random.choice(len(gt_points), subsample, replace=False)
        gt_points = gt_points[idx]

    # Nearest-neighbour distances using broadcasting (memory-efficient for moderate clouds)
    def nn_dist(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        # For large clouds, process in batches to avoid OOM
        batch = 2048
        dists = []
        for i in range(0, len(src), batch):
            d = np.linalg.norm(src[i:i+batch, None] - tgt[None], axis=-1)
            dists.append(d.min(axis=1))
        return np.concatenate(dists)

    acc  = nn_dist(pred_points, gt_points).mean()
    comp = nn_dist(gt_points, pred_points).mean()
    return {
        "accuracy":     float(acc),
        "completeness": float(comp),
        "chamfer":      float((acc + comp) / 2),
    }


# ---------------------------------------------------------------------------
# Memory & timing profilers
# ---------------------------------------------------------------------------

class MemoryProfiler:
    """Context manager that records peak GPU memory usage (MB)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.peak_mb = 0.0
        self._available = False

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
                self._available = True
        except ImportError:
            pass
        return self

    def __exit__(self, *_):
        try:
            import torch
            if self._available:
                self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        except ImportError:
            pass

    @property
    def peak_gb(self) -> float:
        return self.peak_mb / 1024


class Timer:
    """Context manager that records elapsed wall-clock time (seconds)."""

    def __init__(self, sync_cuda: bool = True):
        self._sync = sync_cuda
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        try:
            import torch
            if self._sync and torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self.elapsed = time.perf_counter() - self._start


def benchmark_inference(
    inference_fn,
    n_frames_list: List[int],
    warmup: int = 1,
    repeats: int = 3,
) -> List[Dict]:
    """
    Benchmark inference_fn(n_frames) for each n_frames in n_frames_list.

    inference_fn should accept an integer and run inference, returning any value.

    Returns:
        list of dicts with keys: n_frames, time_mean_s, time_std_s, peak_gpu_mb
    """
    results = []
    for n in n_frames_list:
        times = []
        peak_mb = 0.0
        for rep in range(warmup + repeats):
            with MemoryProfiler() as mem, Timer() as t:
                inference_fn(n)
            if rep >= warmup:
                times.append(t.elapsed)
                peak_mb = max(peak_mb, mem.peak_mb)

        results.append({
            "n_frames":    n,
            "time_mean_s": float(np.mean(times)),
            "time_std_s":  float(np.std(times)),
            "peak_gpu_mb": peak_mb,
        })
        print(f"  n_frames={n:4d}  time={np.mean(times):.2f}s  peak={peak_mb:.0f} MB")
    return results


# ---------------------------------------------------------------------------
# Convenience: print a summary table
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty-print a flat dict of metrics."""
    width = max(len(k) for k in metrics) + 2
    print(f"\n{'─' * (width + 14)}")
    print(f"  {title}")
    print(f"{'─' * (width + 14)}")
    for k, v in metrics.items():
        print(f"  {k:<{width}} {v:.4f}")
    print(f"{'─' * (width + 14)}\n")
