"""
IMU Pre-integration for VGGT + IMU evaluation (Phase 4).

Reads IMU data in EuRoC / TUM-VI format and integrates gyroscope +
accelerometer between consecutive image frames to produce relative-pose
estimates that can constrain or correct VGGT camera predictions.

Key design choices
------------------
- Gyroscope-only integration (`gyro_only_rotations`) is the primary path:
  it avoids accelerometer bias and gravity estimation, giving cleaner
  relative-rotation increments for VGGT fusion.
- Full pre-integration (`integrate_between`) is also available for
  completeness / comparison, but note that position estimates drift quickly
  without proper bias initialisation.
- All maths uses SO(3) Rodrigues' formula and midpoint integration.
  No external dependencies beyond numpy.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues: angular-velocity vector (3,) → rotation matrix (3,3)."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-10:
        return np.eye(3) + _skew(omega)
    axis = omega / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) → angular-velocity vector (3,)."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-10:
        return np.zeros(3)
    axis_skew = (R - R.T) / (2.0 * np.sin(angle))
    axis = np.array([axis_skew[2, 1], axis_skew[0, 2], axis_skew[1, 0]])
    return axis * angle


def slerp_R(R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between rotation matrices.
    t=0 → R1,  t=1 → R2.
    """
    omega = so3_log(R1.T @ R2)
    return R1 @ so3_exp(omega * t)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IMUCalibration:
    """
    IMU noise and bias parameters.

    Defaults are representative values for the TUM-VI dataset
    (BMI160 IMU in the TUM fisheye camera).
    """
    gyro_noise_density: float = 1.6e-4    # rad/s/√Hz
    accel_noise_density: float = 2.0e-3   # m/s²/√Hz
    gyro_random_walk: float = 2.2e-6      # rad/s²/√Hz
    accel_random_walk: float = 4.3e-5     # m/s³/√Hz
    # Constant biases (can be estimated offline from stationary segments)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Rotation from IMU frame to camera frame (identity if co-located)
    R_cam_imu: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_cam_imu: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class IMUReading:
    """One IMU sample."""
    timestamp: float        # seconds
    gyro: np.ndarray        # (3,) rad/s
    accel: np.ndarray       # (3,) m/s²


@dataclass
class PreintegratedIMU:
    """Result of integrating IMU over one inter-frame interval."""
    delta_R: np.ndarray     # (3,3) relative rotation body frame
    delta_v: np.ndarray     # (3,) velocity increment (body frame at start)
    delta_p: np.ndarray     # (3,) position increment (body frame at start)
    dt: float               # total integration time (s)
    n_samples: int = 0      # number of IMU samples used


# ---------------------------------------------------------------------------
# CSV parsing  (EuRoC / TUM-VI format)
# ---------------------------------------------------------------------------

def parse_imu_csv(csv_path: str) -> List[IMUReading]:
    """
    Parse IMU CSV in EuRoC format:

    #timestamp [ns], w_RS_S_x [rad s^-1], w_RS_S_y, w_RS_S_z,
                     a_RS_S_x [m s^-2],   a_RS_S_y, a_RS_S_z

    Returns sorted list of IMUReading objects (timestamps in seconds).
    """
    readings: List[IMUReading] = []
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            ts_s = float(parts[0]) * 1e-9          # ns → s
            gyro  = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            readings.append(IMUReading(timestamp=ts_s, gyro=gyro, accel=accel))

    readings.sort(key=lambda r: r.timestamp)
    return readings


def parse_image_timestamps_csv(csv_path: str) -> List[float]:
    """
    Parse image timestamps from EuRoC data.csv:

    #timestamp [ns], filename
    """
    timestamps: List[float] = []
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            timestamps.append(float(parts[0]) * 1e-9)
    timestamps.sort()
    return timestamps


def parse_groundtruth_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ground-truth CSV in EuRoC format:

    #timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z

    Returns:
        timestamps : (N,) float64 in seconds
        poses      : (N, 7) [p_x, p_y, p_z, q_w, q_x, q_y, q_z]
    """
    ts_list, pose_list = [], []
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            ts_list.append(float(parts[0]) * 1e-9)
            pose_list.append([float(x) for x in parts[1:8]])

    ts    = np.array(ts_list)
    poses = np.array(pose_list)
    order = np.argsort(ts)
    return ts[order], poses[order]


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix
# ---------------------------------------------------------------------------

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [q_w, q_x, q_y, q_z] to 3×3 rotation matrix.
    """
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),  1 - 2*(qx*qx + qy*qy)],
    ])


def interpolate_groundtruth(
    gt_timestamps: np.ndarray,
    gt_poses: np.ndarray,
    query_timestamps: List[float],
) -> np.ndarray:
    """
    Linearly interpolate ground-truth poses to image frame timestamps.

    gt_poses: (N, 7) [p_x, p_y, p_z, q_w, q_x, q_y, q_z]
    Returns:  (M, 3, 4) extrinsic matrices [R | t] for each query timestamp.
    The convention is world-to-body (IMU/body frame).
    """
    extrinsics = []
    for t in query_timestamps:
        idx = np.searchsorted(gt_timestamps, t)
        idx = np.clip(idx, 1, len(gt_timestamps) - 1)
        t0, t1 = gt_timestamps[idx - 1], gt_timestamps[idx]
        alpha = (t - t0) / (t1 - t0 + 1e-12)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        p0, p1 = gt_poses[idx - 1, :3], gt_poses[idx, :3]
        q0, q1 = gt_poses[idx - 1, 3:], gt_poses[idx, 3:]

        p = (1 - alpha) * p0 + alpha * p1

        # Quaternion slerp
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1 = -q1
            dot = -dot
        dot = min(dot, 1.0)
        theta = float(np.arccos(dot))
        if abs(theta) < 1e-8:
            q = q0
        else:
            q = (np.sin((1 - alpha) * theta) * q0 +
                 np.sin(alpha * theta) * q1) / np.sin(theta)
        q = q / np.linalg.norm(q)

        R = quat_to_R(q)
        # world-to-body extrinsic: E = [R | -R @ p]
        # (body frame is the IMU/camera body; p is position in world)
        E = np.zeros((3, 4))
        E[:3, :3] = R.T        # world-to-body rotation
        E[:3,  3] = p          # position in world frame
        extrinsics.append(E)

    return np.stack(extrinsics)


# ---------------------------------------------------------------------------
# IMU Pre-integrator
# ---------------------------------------------------------------------------

class IMUPreintegrator:
    """
    Integrates IMU readings between image frame timestamps.

    Two modes
    ---------
    gyro_only_rotations   : integrate only gyroscope → relative rotations.
                            Robust, no gravity/bias issues.
    integrate_between     : full pre-integration (R, v, p) for one interval.
    integrate_all_frames  : calls integrate_between for each image pair.
    """

    def __init__(
        self,
        calibration: Optional[IMUCalibration] = None,
        gravity: Optional[np.ndarray] = None,
    ):
        self.calib   = calibration or IMUCalibration()
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

    # ------------------------------------------------------------------
    def integrate_between(
        self,
        imu_readings: List[IMUReading],
        t_start: float,
        t_end: float,
    ) -> PreintegratedIMU:
        """
        Full mid-point pre-integration between t_start and t_end.

        Gravity is NOT subtracted here; caller must account for it when
        reconstructing absolute positions.
        """
        g_bias = self.calib.gyro_bias
        a_bias = self.calib.accel_bias

        # Filter to [t_start, t_end]
        samples = [r for r in imu_readings if t_start <= r.timestamp <= t_end]
        if len(samples) < 2:
            return PreintegratedIMU(
                delta_R=np.eye(3), delta_v=np.zeros(3),
                delta_p=np.zeros(3), dt=t_end - t_start, n_samples=len(samples),
            )

        delta_R = np.eye(3)
        delta_v = np.zeros(3)
        delta_p = np.zeros(3)

        prev_t     = samples[0].timestamp
        prev_gyro  = samples[0].gyro  - g_bias
        prev_accel = samples[0].accel - a_bias

        for s in samples[1:]:
            dt = s.timestamp - prev_t
            if dt <= 0.0:
                continue

            curr_gyro  = s.gyro  - g_bias
            curr_accel = s.accel - a_bias

            # Mid-point averages
            mid_gyro  = 0.5 * (prev_gyro  + curr_gyro)
            mid_accel = 0.5 * (prev_accel + curr_accel)

            # Rotate mid_accel to body-start frame, then accumulate
            acc_body = delta_R @ mid_accel
            delta_p += delta_v * dt + 0.5 * acc_body * dt * dt
            delta_v += acc_body * dt
            delta_R  = delta_R @ so3_exp(mid_gyro * dt)

            prev_t     = s.timestamp
            prev_gyro  = curr_gyro
            prev_accel = curr_accel

        return PreintegratedIMU(
            delta_R=delta_R, delta_v=delta_v, delta_p=delta_p,
            dt=t_end - t_start, n_samples=len(samples),
        )

    # ------------------------------------------------------------------
    def integrate_all_frames(
        self,
        imu_readings: List[IMUReading],
        image_timestamps: List[float],
    ) -> List[PreintegratedIMU]:
        """
        Integrate IMU between each consecutive image-frame pair.

        Returns list of length len(image_timestamps)-1.
        """
        return [
            self.integrate_between(imu_readings, image_timestamps[i], image_timestamps[i + 1])
            for i in range(len(image_timestamps) - 1)
        ]

    # ------------------------------------------------------------------
    def gyro_only_rotations(
        self,
        imu_readings: List[IMUReading],
        image_timestamps: List[float],
        R0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Integrate only the gyroscope to get one absolute rotation per frame.

        Much more reliable than full integration because:
        - No accelerometer bias / gravity estimation needed
        - Gyroscope drift is slow (minutes), negligible over 2–5 s sequences

        Args:
            imu_readings      : all IMU readings for the sequence
            image_timestamps  : timestamps for each selected image frame (s)
            R0                : initial rotation (3,3), default = identity

        Returns:
            rotations (N, 3, 3) — one absolute rotation matrix per frame
        """
        if R0 is None:
            R0 = np.eye(3)

        g_bias = self.calib.gyro_bias
        N = len(image_timestamps)
        rotations = np.zeros((N, 3, 3))
        rotations[0] = R0

        # Build a timestamp-indexed lookup into imu_readings for efficiency
        imu_ts = np.array([r.timestamp for r in imu_readings])

        for i in range(N - 1):
            t0, t1 = image_timestamps[i], image_timestamps[i + 1]
            lo = int(np.searchsorted(imu_ts, t0, side="left"))
            hi = int(np.searchsorted(imu_ts, t1, side="right"))
            samples = imu_readings[lo:hi]

            R = rotations[i].copy()
            if len(samples) < 2:
                rotations[i + 1] = R
                continue

            prev_t    = samples[0].timestamp
            prev_gyro = samples[0].gyro - g_bias
            for s in samples[1:]:
                dt = s.timestamp - prev_t
                if dt <= 0.0:
                    continue
                curr_gyro = s.gyro - g_bias
                mid_gyro  = 0.5 * (prev_gyro + curr_gyro)
                R = R @ so3_exp(mid_gyro * dt)
                prev_t    = s.timestamp
                prev_gyro = curr_gyro

            rotations[i + 1] = R

        return rotations


# ---------------------------------------------------------------------------
# Utility: estimate gyro bias from stationary segment at start of sequence
# ---------------------------------------------------------------------------

def estimate_gyro_bias(
    imu_readings: List[IMUReading],
    duration_s: float = 1.0,
) -> np.ndarray:
    """
    Estimate gyroscope bias as the mean of readings in the first `duration_s`
    seconds (assuming the sensor is stationary during that period).

    Returns: (3,) bias vector in rad/s
    """
    if not imu_readings:
        return np.zeros(3)
    t0 = imu_readings[0].timestamp
    stationary = [r for r in imu_readings if r.timestamp - t0 <= duration_s]
    if not stationary:
        return np.zeros(3)
    return np.mean([r.gyro for r in stationary], axis=0)
