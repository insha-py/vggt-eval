"""
TUM Visual-Inertial Dataset loader for VGGT evaluation.

Dataset page: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
Format: EuRoC MAV  (same as ETH ASL datasets)

Extracted layout
----------------
{sequence_name}/
  mav0/
    cam0/
      data/         <- PNG images, filename = timestamp in ns
      data.csv      <- #timestamp [ns], filename
      sensor.yaml   <- camera intrinsics + extrinsics
    imu0/
      data.csv      <- #timestamp [ns], wx, wy, wz, ax, ay, az
    mocap0/
      data.csv      <- #timestamp [ns], px, py, pz, qw, qx, qy, qz

Available sequences (we prefer the short indoor "room" ones):
  room1 … room6
  corridor1 … corridor5
  magistrale1 … magistrale6   (long, skip for this eval)

Usage
-----
from src.tum_vi import TUMVIDataset

ds = TUMVIDataset(sequence="room1", n_frames=40, download_dir="/tmp/tumvi")
ds.download()          # idempotent; skips if already extracted
data = ds.load()
# data keys:
#   images_np       (N, 3, H, W) float32 in [0,1]
#   image_paths     list[str]
#   image_timestamps list[float] (seconds)
#   imu_readings    list[IMUReading]
#   gt_extrinsics   (N, 3, 4)  world-to-body (interpolated to image times)
#   gt_timestamps   np.ndarray (all GT timestamps, seconds)
#   calib           IMUCalibration
"""

from __future__ import annotations

import os
import tarfile
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.imu import (
    IMUCalibration,
    IMUReading,
    parse_imu_csv,
    parse_image_timestamps_csv,
    parse_groundtruth_csv,
    interpolate_groundtruth,
)


# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

# Base URL for the 512-px 16-fps EuRoC export
_BASE_URL = "https://cvg.cit.tum.de/tumvi/exported/euroc/512_16"

# Sequences suitable for < 50 frame evaluations (all ≤ ~35 s @ 20 fps)
AVAILABLE_SEQUENCES = [
    "room1", "room2", "room3", "room4", "room5", "room6",
    "corridor1", "corridor2", "corridor3", "corridor4", "corridor5",
    "slides1", "slides2", "slides3",
]

# TUM VI calibration for the 512-px export (cam0, pinhole model)
# fx, fy, cx, cy in pixels (approximate; sensor.yaml has exact values)
_DEFAULT_CALIB = {
    "room":      dict(fx=190.97847715128717, fy=190.9733070521226,
                      cx=254.93170605935475, cy=256.8974428996504),
    "corridor":  dict(fx=190.97847715128717, fy=190.9733070521226,
                      cx=254.93170605935475, cy=256.8974428996504),
    "slides":    dict(fx=190.97847715128717, fy=190.9733070521226,
                      cx=254.93170605935475, cy=256.8974428996504),
}

# IMU-to-camera calibration (rotation) for TUM VI
# The camera is mounted with a known rotation relative to IMU body frame.
# These values come from the official calibration files.
_R_CAM_IMU = np.array([
    [ 0.0148655429818, -0.999880929698,  0.00414029679422],
    [ 0.999557249008,   0.0149672133247, 0.025715529948  ],
    [-0.0257744366974,  0.00375618835797, 0.999660727178 ],
])


# ---------------------------------------------------------------------------
# TUMVIDataset
# ---------------------------------------------------------------------------

class TUMVIDataset:
    """
    Downloads, extracts, and loads a TUM VI sequence for VGGT evaluation.

    Args:
        sequence      : one of AVAILABLE_SEQUENCES, e.g. "room1"
        n_frames      : how many frames to return (≤ 50 recommended)
        download_dir  : where to store the downloaded/extracted data
        frame_stride  : if set, sample every Nth image instead of uniform
    """

    def __init__(
        self,
        sequence: str = "room1",
        n_frames: int = 40,
        download_dir: str = "/tmp/tumvi",
        frame_stride: Optional[int] = None,
    ):
        if sequence not in AVAILABLE_SEQUENCES:
            raise ValueError(
                f"Unknown sequence '{sequence}'. "
                f"Choose from: {AVAILABLE_SEQUENCES}"
            )
        self.sequence     = sequence
        self.n_frames     = n_frames
        self.download_dir = Path(download_dir)
        self.frame_stride = frame_stride

        # Derived paths
        self._tar_name = f"dataset-{sequence}_512_16.tar"
        self._url      = f"{_BASE_URL}/{self._tar_name}"
        self._root     = self.download_dir / f"dataset-{sequence}_512_16" / "mav0"

    # ------------------------------------------------------------------
    # Properties for sub-paths
    # ------------------------------------------------------------------
    @property
    def cam0_dir(self) -> Path:
        return self._root / "cam0" / "data"

    @property
    def cam0_csv(self) -> Path:
        return self._root / "cam0" / "data.csv"

    @property
    def imu_csv(self) -> Path:
        return self._root / "imu0" / "data.csv"

    @property
    def mocap_csv(self) -> Path:
        return self._root / "mocap0" / "data.csv"

    # ------------------------------------------------------------------
    def is_downloaded(self) -> bool:
        return self.cam0_dir.exists() and any(self.cam0_dir.glob("*.png"))

    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> None:
        """
        Download and extract the dataset if not already present.

        Set force=True to re-download even if data exists.
        """
        if self.is_downloaded() and not force:
            print(f"[TUM-VI] {self.sequence} already extracted at {self._root}")
            return

        self.download_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.download_dir / self._tar_name

        if not tar_path.exists() or force:
            print(f"[TUM-VI] Downloading {self._url} ...")
            print(f"[TUM-VI] This may take several minutes (file is ~400 MB).")

            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, 100 * downloaded / total_size)
                    mb  = downloaded / 1024**2
                    print(f"\r[TUM-VI]   {pct:.1f}%  ({mb:.0f} MB)", end="", flush=True)

            urllib.request.urlretrieve(self._url, tar_path, reporthook=_progress)
            print()  # newline after progress

        print(f"[TUM-VI] Extracting {tar_path} …")
        with tarfile.open(tar_path) as tf:
            tf.extractall(self.download_dir)
        print(f"[TUM-VI] Extracted to {self.download_dir}")

    # ------------------------------------------------------------------
    def _select_frame_indices(self, total: int) -> List[int]:
        """Pick n_frames indices evenly spaced from [0, total)."""
        if self.frame_stride is not None:
            indices = list(range(0, total, self.frame_stride))[: self.n_frames]
        else:
            indices = np.linspace(0, total - 1, min(self.n_frames, total), dtype=int).tolist()
        return [int(i) for i in indices]

    # ------------------------------------------------------------------
    def load(self) -> Dict:
        """
        Load the dataset into memory.

        Returns a dict with keys:
            images_np       : (N, 3, H, W) float32  [0,1]
            image_paths     : list[str]
            image_timestamps: list[float] in seconds
            imu_readings    : list[IMUReading]
            gt_extrinsics   : (N, 3, 4) world-to-body at image timestamps
            gt_positions    : (N, 3)   world-frame positions
            gt_ts_all       : (M,) all ground-truth timestamps
            calib           : IMUCalibration
        """
        if not self.is_downloaded():
            raise RuntimeError(
                f"Dataset not found at {self._root}. Call download() first."
            )

        from PIL import Image as PILImage
        from torchvision import transforms as TF

        # ---- image timestamps ----------------------------------------
        all_ts = parse_image_timestamps_csv(str(self.cam0_csv))
        all_img_paths = sorted(self.cam0_dir.glob("*.png"),
                               key=lambda p: p.name)

        # Align: match CSV rows to actual files
        # (sometimes the CSV has extra/missing entries; use intersection)
        img_map = {p.stem: str(p) for p in all_img_paths}
        ts_path_pairs = []
        for ts in all_ts:
            stem = str(int(round(ts * 1e9)))
            if stem in img_map:
                ts_path_pairs.append((ts, img_map[stem]))
        ts_path_pairs.sort(key=lambda x: x[0])

        total = len(ts_path_pairs)
        if total == 0:
            raise RuntimeError(f"No matching images found in {self.cam0_dir}")

        sel_idx   = self._select_frame_indices(total)
        sel_pairs = [ts_path_pairs[i] for i in sel_idx]
        image_timestamps = [p[0] for p in sel_pairs]
        image_paths      = [p[1] for p in sel_pairs]
        N = len(image_paths)
        print(f"[TUM-VI] Selected {N}/{total} frames from '{self.sequence}'")

        # ---- load images ---------------------------------------------
        to_tensor = TF.ToTensor()
        tensors = []
        for path in image_paths:
            img = PILImage.open(path).convert("RGB")
            tensors.append(to_tensor(img))
        images_np = np.stack([t.numpy() for t in tensors])  # (N, 3, H, W)
        print(f"[TUM-VI] Images loaded: {images_np.shape}")

        # ---- IMU data ------------------------------------------------
        imu_readings = parse_imu_csv(str(self.imu_csv))
        print(f"[TUM-VI] IMU readings: {len(imu_readings)}")

        # ---- Ground truth -------------------------------------------
        gt_ts, gt_poses = parse_groundtruth_csv(str(self.mocap_csv))
        gt_extrinsics = interpolate_groundtruth(gt_ts, gt_poses, image_timestamps)
        gt_positions  = gt_poses[:, :3]  # all positions for reference
        print(f"[TUM-VI] GT loaded ({len(gt_ts)} poses)")

        # ---- Calibration (use defaults for 512-px TUM-VI export) -----
        prefix = "room" if self.sequence.startswith("room") else \
                 "corridor" if self.sequence.startswith("corridor") else "slides"
        cam_cal = _DEFAULT_CALIB.get(prefix, _DEFAULT_CALIB["room"])
        calib = IMUCalibration(R_cam_imu=_R_CAM_IMU)

        return dict(
            images_np        = images_np,
            image_paths      = image_paths,
            image_timestamps = image_timestamps,
            imu_readings     = imu_readings,
            gt_extrinsics    = gt_extrinsics,
            gt_positions     = gt_positions,
            gt_ts_all        = gt_ts,
            calib            = calib,
            cam_intrinsics   = cam_cal,
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"TUMVIDataset(sequence={self.sequence!r}, "
                f"n_frames={self.n_frames}, root={self._root})")
