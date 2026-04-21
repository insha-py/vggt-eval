"""
IMU + VGGT pose fusion (Phase 4).

Strategy
--------
VGGT predicts absolute camera poses (world-to-camera extrinsics) from images
alone. IMU gyroscope integration gives us reliable *relative* rotations between
consecutive frames (short-term, free of drift over 2–5 s windows).

We fuse them at the *relative-rotation* level:

  1. Extract relative rotations from VGGT:   R_rel_vggt[i] = R_v[i+1] @ R_v[i]^T
  2. Integrate IMU gyroscope for each interval → R_rel_imu[i]
  3. Align IMU to camera frame via the known R_cam_imu calibration.
  4. SLERP-blend: R_rel_fused[i] = slerp(R_rel_vggt[i], R_rel_imu_aligned[i], alpha)
     alpha=0 → pure VGGT,  alpha=1 → pure IMU
  5. Reconstruct absolute rotations by chaining from R_v[0] (first VGGT pose).
  6. Keep camera *centres* from VGGT (translation), just swap in the fused rotation.
     This preserves VGGT's metrically-consistent scale without trying to integrate
     accelerometers (which requires accurate gravity+bias estimation).

The result is an (N, 3, 4) extrinsics array that blends VGGT's pose quality
with the IMU's short-term rotational smoothness.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.imu import (
    IMUCalibration,
    IMUPreintegrator,
    IMUReading,
    slerp_R,
    so3_exp,
    so3_log,
)


# ---------------------------------------------------------------------------
# IMUVGGTFusion
# ---------------------------------------------------------------------------

class IMUVGGTFusion:
    """
    Fuse VGGT extrinsic predictions with IMU gyroscope integration.

    Args:
        alpha      : blend weight for IMU (0 = pure VGGT, 1 = pure IMU).
        calibration: IMU calibration (noise params + R_cam_imu).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        calibration: Optional[IMUCalibration] = None,
    ):
        self.alpha   = float(np.clip(alpha, 0.0, 1.0))
        self.calib   = calibration or IMUCalibration()
        self._preint = IMUPreintegrator(calibration=self.calib)

    # ------------------------------------------------------------------
    def fuse(
        self,
        vggt_extrinsics: np.ndarray,        # (N, 3, 4)
        imu_readings: List[IMUReading],
        image_timestamps: List[float],      # length N
    ) -> np.ndarray:
        """
        Fuse VGGT extrinsics with IMU rotations.

        Returns: (N, 3, 4) fused extrinsics (same format as VGGT output).
        """
        N = len(vggt_extrinsics)
        assert len(image_timestamps) == N, "timestamps must match extrinsic count"

        R_vggt = vggt_extrinsics[:, :3, :3]   # (N, 3, 3)
        t_vggt = vggt_extrinsics[:, :3,  3]   # (N, 3)

        # ---- IMU gyro-only absolute rotations (one per frame) ---------
        R_imu = self._preint.gyro_only_rotations(
            imu_readings, image_timestamps, R0=np.eye(3)
        )  # (N, 3, 3)  — in IMU body frame

        # ---- Align IMU frame to VGGT camera frame ---------------------
        # R_cam_imu rotates vectors from IMU to camera frame.
        # For relative rotations the conversion is:
        #   R_rel_cam = R_cam_imu @ R_rel_imu @ R_cam_imu^T
        R_ci = self.calib.R_cam_imu  # (3, 3)

        R_rel_imu = np.array([
            R_imu[i + 1] @ R_imu[i].T
            for i in range(N - 1)
        ])  # (N-1, 3, 3) — incremental IMU body rotations

        R_rel_imu_cam = np.array([
            R_ci @ R_rel_imu[i] @ R_ci.T
            for i in range(N - 1)
        ])  # (N-1, 3, 3) — now in camera frame

        # ---- VGGT relative rotations ---------------------------------
        R_rel_vggt = np.array([
            R_vggt[i + 1] @ R_vggt[i].T
            for i in range(N - 1)
        ])  # (N-1, 3, 3)

        # ---- SLERP blend at the relative-rotation level --------------
        R_rel_fused = np.array([
            slerp_R(R_rel_vggt[i], R_rel_imu_cam[i], self.alpha)
            for i in range(N - 1)
        ])  # (N-1, 3, 3)

        # ---- Reconstruct absolute rotations from R_v[0] ---------------
        R_fused = np.zeros((N, 3, 3))
        R_fused[0] = R_vggt[0]
        for i in range(N - 1):
            R_fused[i + 1] = R_rel_fused[i] @ R_fused[i]

        # ---- Keep camera centres from VGGT, update rotation ----------
        # Camera centre: c = -R^T @ t
        # New translation: t_new = -R_fused @ c_vggt
        c_vggt = -np.einsum("nij,nj->ni", R_vggt.transpose(0, 2, 1), t_vggt)  # (N,3)
        t_fused = -np.einsum("nij,nj->ni", R_fused, c_vggt)                   # (N,3)

        out = np.zeros((N, 3, 4))
        out[:, :3, :3] = R_fused
        out[:, :3,  3] = t_fused
        return out

    # ------------------------------------------------------------------
    def fuse_from_preintegrated(
        self,
        vggt_extrinsics: np.ndarray,        # (N, 3, 4)
        R_imu_abs: np.ndarray,              # (N, 3, 3) pre-computed IMU rotations
    ) -> np.ndarray:
        """
        Fuse using pre-computed IMU absolute rotations (e.g. from a VIO system).
        Useful when you already have IMU rotations from an external source.

        R_imu_abs: absolute rotations in IMU body frame (identity at frame 0).
        """
        N = len(vggt_extrinsics)
        R_vggt = vggt_extrinsics[:, :3, :3]
        t_vggt = vggt_extrinsics[:, :3,  3]
        R_ci   = self.calib.R_cam_imu

        R_rel_imu_cam = np.array([
            R_ci @ (R_imu_abs[i + 1] @ R_imu_abs[i].T) @ R_ci.T
            for i in range(N - 1)
        ])
        R_rel_vggt = np.array([
            R_vggt[i + 1] @ R_vggt[i].T
            for i in range(N - 1)
        ])
        R_rel_fused = np.array([
            slerp_R(R_rel_vggt[i], R_rel_imu_cam[i], self.alpha)
            for i in range(N - 1)
        ])

        R_fused = np.zeros((N, 3, 3))
        R_fused[0] = R_vggt[0]
        for i in range(N - 1):
            R_fused[i + 1] = R_rel_fused[i] @ R_fused[i]

        c_vggt  = -np.einsum("nij,nj->ni", R_vggt.transpose(0, 2, 1), t_vggt)
        t_fused = -np.einsum("nij,nj->ni", R_fused, c_vggt)

        out = np.zeros((N, 3, 4))
        out[:, :3, :3] = R_fused
        out[:, :3,  3] = t_fused
        return out


# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------

def alpha_sweep(
    vggt_extrinsics: np.ndarray,
    imu_readings: List[IMUReading],
    image_timestamps: List[float],
    gt_extrinsics: np.ndarray,
    alphas: List[float],
    calibration: Optional[IMUCalibration] = None,
) -> List[dict]:
    """
    Run fusion at multiple alpha values and compute ATE for each.

    Returns list of dicts: {alpha, ate_mean, ate_rmse, ate_median}.
    """
    from src.metrics import compute_ate, compute_rpe

    results = []
    for alpha in alphas:
        fuser = IMUVGGTFusion(alpha=alpha, calibration=calibration)
        fused_ext = fuser.fuse(vggt_extrinsics, imu_readings, image_timestamps)

        ate  = compute_ate(fused_ext, gt_extrinsics, align=True, with_scale=True)
        rpe  = compute_rpe(fused_ext, gt_extrinsics, step=1)

        results.append(dict(
            alpha      = alpha,
            ate_mean   = ate["mean"],
            ate_rmse   = ate["rmse"],
            ate_median = ate["median"],
            rpe_trans  = rpe["trans_mean"],
            rpe_rot    = rpe["rot_mean"],
        ))
        print(f"  alpha={alpha:.2f}  ATE={ate['mean']:.4f}  "
              f"RPE_t={rpe['trans_mean']:.4f}  RPE_r={rpe['rot_mean']:.2f}°")

    return results
