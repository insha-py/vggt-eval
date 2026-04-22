"""
TUM Visual-Inertial Dataset loader for VGGT evaluation.

Dataset page: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
Format: EuRoC MAV

Download strategy
-----------------
The full tar archives are 1–2 GB each. We avoid downloading the whole thing
by using HTTP Range requests to scan tar file headers (512 B each), build an
in-memory index, and then fetch only the files we actually need:
  - mav0/cam0/data.csv        (image timestamps)
  - mav0/imu0/data.csv        (IMU readings)
  - mav0/mocap0/data.csv      (ground-truth poses)
  - first N images from mav0/cam0/data/*.png

Total transferred: typically 10–25 MB instead of 1.6 GB.
Falls back to a streaming tar extraction if the server does not advertise
Accept-Ranges: bytes.
"""

from __future__ import annotations

import os
import io
import tarfile
import struct
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

_BASE_URL = "https://cvg.cit.tum.de/tumvi/exported/euroc/512_16"

AVAILABLE_SEQUENCES = [
    "room1", "room2", "room3", "room4", "room5", "room6",
    "corridor1", "corridor2", "corridor3", "corridor4", "corridor5",
    "slides1", "slides2", "slides3",
]

_DEFAULT_CALIB = {
    "room":     dict(fx=190.97847715128717, fy=190.9733070521226,
                     cx=254.93170605935475, cy=256.8974428996504),
    "corridor": dict(fx=190.97847715128717, fy=190.9733070521226,
                     cx=254.93170605935475, cy=256.8974428996504),
    "slides":   dict(fx=190.97847715128717, fy=190.9733070521226,
                     cx=254.93170605935475, cy=256.8974428996504),
}

_R_CAM_IMU = np.array([
    [ 0.0148655429818, -0.999880929698,  0.00414029679422],
    [ 0.999557249008,   0.0149672133247, 0.025715529948  ],
    [-0.0257744366974,  0.00375618835797, 0.999660727178 ],
])

# CSVs we always need (relative to mav0/)
_REQUIRED_CSVS = {"cam0/data.csv", "imu0/data.csv", "mocap0/data.csv"}


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _http_head(url: str) -> dict:
    """Return response headers for a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return dict(resp.headers)


def _http_range(url: str, start: int, end: int) -> bytes:
    """
    Download bytes [start, end] (inclusive) from url using HTTP Range.
    """
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Tar index builder (Range-based, downloads only 512-byte headers)
# ---------------------------------------------------------------------------

def _parse_tar_header(block: bytes):
    """
    Parse one 512-byte POSIX tar header block.
    Returns (name, size_bytes) or None if end-of-archive or invalid.
    """
    if len(block) < 512 or block[:100] == b"\x00" * 100:
        return None
    try:
        name = block[:100].rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        # Long-name (GNU/POSIX extension) prefix stored in bytes 345-499
        prefix = block[345:500].rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        if prefix:
            name = prefix + "/" + name
        size_field = block[124:136].rstrip(b"\x00")
        size = int(size_field or b"0", 8) if size_field else 0
        typeflag = chr(block[156]) if block[156] else "0"
    except Exception:
        return None
    return name, size, typeflag


def build_tar_index(url: str, total_size: int) -> Dict[str, Tuple[int, int]]:
    """
    Scan a remote tar by fetching only the 512-byte header of each member
    using HTTP Range requests.

    Returns {member_name: (data_offset, data_size_bytes)}.

    Header reads: O(N) round-trips where N = number of tar members.
    For a 1600-image archive this is ~1600 requests × 512 B ≈ 0.8 MB.
    """
    index: Dict[str, Tuple[int, int]] = {}
    offset = 0
    consecutive_nulls = 0

    while offset + 512 <= total_size:
        try:
            block = _http_range(url, offset, offset + 511)
        except Exception as e:
            print(f"[TUM-VI]   Range read failed at offset {offset}: {e}")
            break

        if len(block) < 512:
            break

        result = _parse_tar_header(block)
        if result is None:
            consecutive_nulls += 1
            if consecutive_nulls >= 2:
                break
            offset += 512
            continue

        consecutive_nulls = 0
        name, size, typeflag = result

        data_offset = offset + 512
        if typeflag in ("0", "\x00", ""):  # regular file
            index[name] = (data_offset, size)

        # Advance past this member (data padded to 512-byte blocks)
        blocks = (size + 511) // 512
        offset = data_offset + blocks * 512

    return index


# ---------------------------------------------------------------------------
# TUMVIDataset
# ---------------------------------------------------------------------------

class TUMVIDataset:
    """
    Downloads and loads a TUM VI sequence for VGGT evaluation.

    Uses HTTP Range requests to download only the needed files (~10–25 MB)
    instead of the full archive (1–2 GB).

    Args:
        sequence     : one of AVAILABLE_SEQUENCES, e.g. "room1"
        n_frames     : number of frames to use (≤ 50 recommended)
        download_dir : where to store extracted files
    """

    def __init__(
        self,
        sequence: str = "room1",
        n_frames: int = 40,
        download_dir: str = "/tmp/tumvi",
    ):
        if sequence not in AVAILABLE_SEQUENCES:
            raise ValueError(
                f"Unknown sequence '{sequence}'. "
                f"Choose from: {AVAILABLE_SEQUENCES}"
            )
        self.sequence     = sequence
        self.n_frames     = n_frames
        self.download_dir = Path(download_dir)

        self._tar_name = f"dataset-{sequence}_512_16.tar"
        self._url      = f"{_BASE_URL}/{self._tar_name}"
        # Extracted root: download_dir / dataset-room1_512_16 / mav0
        self._root = self.download_dir / f"dataset-{sequence}_512_16" / "mav0"

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

    def _csvs_present(self) -> bool:
        return self.cam0_csv.exists() and self.imu_csv.exists() and self.mocap_csv.exists()

    def is_downloaded(self) -> bool:
        return self._csvs_present() and any(self.cam0_dir.glob("*.png"))

    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> None:
        """
        Download only the files we need via HTTP Range requests.
        Falls back to streaming tar if the server doesn't support ranges.
        """
        if self.is_downloaded() and not force:
            n_imgs = len(list(self.cam0_dir.glob("*.png")))
            print(f"[TUM-VI] '{self.sequence}' already present "
                  f"({n_imgs} images, CSVs OK)")
            return

        self.download_dir.mkdir(parents=True, exist_ok=True)

        # --- Check Range support ---
        try:
            headers = _http_head(self._url)
            total_size = int(headers.get("Content-Length", 0))
            accepts_ranges = headers.get("Accept-Ranges", "").lower() == "bytes"
        except Exception as e:
            print(f"[TUM-VI] HEAD request failed ({e}); falling back to streaming.")
            accepts_ranges = False
            total_size = 0

        if accepts_ranges and total_size > 0:
            success = self._download_selective(total_size)
            if success:
                return
            print("[TUM-VI] Selective download failed, falling back to streaming …")

        self._download_streaming()

    # ------------------------------------------------------------------
    def _download_selective(self, total_size: int) -> bool:
        """
        Build tar index via Range header scans, then fetch only needed files.
        """
        print(f"[TUM-VI] Scanning tar index for '{self.sequence}' "
              f"({total_size / 1024**2:.0f} MB archive) …")
        print(f"[TUM-VI] This downloads only ~512 B per file entry — "
              f"much faster than the full archive.")

        try:
            index = build_tar_index(self._url, total_size)
        except Exception as e:
            print(f"[TUM-VI] Index scan error: {e}")
            return False

        if not index:
            print("[TUM-VI] Empty index — tar may use an unsupported format.")
            return False

        print(f"[TUM-VI] Index built: {len(index)} entries found.")

        # --- Identify needed files ---
        # Strip the top-level directory prefix (dataset-room1_512_16/mav0/...)
        def rel_path(name: str) -> str:
            parts = name.lstrip("/").split("/")
            # Drop the first component (dataset-room1_512_16)
            return "/".join(parts[1:]) if len(parts) > 1 else name

        # Sort images by name (= by timestamp) so "first N" = earliest N frames
        image_entries = sorted(
            [(rel_path(n), off, sz) for n, (off, sz) in index.items()
             if "cam0/data/" in n and n.endswith(".png")],
            key=lambda x: x[0],
        )

        # Match CSVs regardless of how many prefix components were stripped.
        # endswith() handles both "mav0/cam0/data.csv" and "cam0/data.csv".
        csv_entries = [
            (rel_path(n), off, sz) for n, (off, sz) in index.items()
            if any(rel_path(n).endswith(c) for c in _REQUIRED_CSVS)
        ]

        selected_images = image_entries[: self.n_frames]

        if not selected_images:
            print("[TUM-VI] No images found in tar index.")
            return False
        if len(csv_entries) < 3:
            print(f"[TUM-VI] Only {len(csv_entries)}/3 CSVs found in index; "
                  f"falling back.")
            return False

        needed = csv_entries + selected_images
        total_bytes = sum(sz for _, _, sz in needed)
        print(f"[TUM-VI] Fetching {len(needed)} files "
              f"({total_bytes / 1024**2:.1f} MB) …")

        # --- Download each file ---
        for i, (rel, offset, size) in enumerate(needed):
            # Reconstruct output path: download_dir / dataset-xxx / rel
            out_path = self.download_dir / f"dataset-{self.sequence}_512_16" / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and out_path.stat().st_size == size:
                continue  # already have it

            try:
                data = _http_range(self._url, offset, offset + size - 1)
            except Exception as e:
                print(f"\n[TUM-VI] Failed to fetch {rel}: {e}")
                return False

            out_path.write_bytes(data)

            mb_done = sum(
                needed[j][2] for j in range(i + 1)
            ) / 1024**2
            print(f"\r[TUM-VI]   {i+1}/{len(needed)} files  "
                  f"({mb_done:.1f} MB)", end="", flush=True)

        print(f"\n[TUM-VI] Done — {len(selected_images)} images + 3 CSVs saved.")
        return True

    # ------------------------------------------------------------------
    def _download_streaming(self) -> None:
        """
        Fallback: stream the tar and extract only what we need.
        We extract first N images + CSVs, then close the connection.
        Still much less disk usage than saving the full tar.
        """
        print(f"[TUM-VI] Streaming tar from {self._url} …")
        print(f"[TUM-VI] Will stop after {self.n_frames} images + 3 CSVs.")

        images_saved = 0
        csvs_saved: set = set()
        bytes_read = 0

        def rel_path(name: str) -> str:
            parts = name.lstrip("/").split("/")
            return "/".join(parts[1:]) if len(parts) > 1 else name

        try:
            with urllib.request.urlopen(self._url, timeout=120) as resp:
                with tarfile.open(fileobj=resp, mode="r|") as tf:
                    for member in tf:
                        rel = rel_path(member.name)

                        is_csv = any(rel.endswith(c) for c in _REQUIRED_CSVS)
                        is_img = ("cam0/data/" in rel and rel.endswith(".png")
                                  and images_saved < self.n_frames)

                        if is_csv or is_img:
                            # rel has the top-level dir stripped, so the
                            # correct output path is download_dir/dataset-xxx/rel
                            out = (self.download_dir
                                   / f"dataset-{self.sequence}_512_16"
                                   / rel)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            fobj = tf.extractfile(member)
                            if fobj:
                                out.write_bytes(fobj.read())
                            if is_img:
                                images_saved += 1
                            if is_csv:
                                csvs_saved.add(rel)

                        bytes_read += member.size
                        mb = bytes_read / 1024**2
                        print(f"\r[TUM-VI]   {mb:.0f} MB read  "
                              f"| images={images_saved}/{self.n_frames}  "
                              f"| csvs={len(csvs_saved)}/3",
                              end="", flush=True)

                        if images_saved >= self.n_frames and len(csvs_saved) >= 3:
                            print(f"\n[TUM-VI] Got everything needed, closing stream.")
                            break

        except Exception as e:
            if images_saved > 0:
                print(f"\n[TUM-VI] Stream ended ({e}); "
                      f"saved {images_saved} images, {len(csvs_saved)} CSVs.")
            else:
                raise

    # ------------------------------------------------------------------
    def _select_frame_indices(self, total: int) -> List[int]:
        return np.linspace(0, total - 1, min(self.n_frames, total),
                           dtype=int).tolist()

    # ------------------------------------------------------------------
    def load(self) -> Dict:
        """
        Load dataset into memory.

        Returns dict with keys:
            images_np, image_paths, image_timestamps,
            imu_readings, gt_extrinsics, gt_positions,
            gt_ts_all, calib, cam_intrinsics
        """
        if not self.is_downloaded():
            raise RuntimeError(
                f"Dataset not ready at {self._root}. Call download() first."
            )

        from PIL import Image as PILImage
        from torchvision import transforms as TF

        # ---- image timestamps & paths ----------------------------------
        all_ts     = parse_image_timestamps_csv(str(self.cam0_csv))
        all_paths  = sorted(self.cam0_dir.glob("*.png"), key=lambda p: p.name)

        img_map = {p.stem: str(p) for p in all_paths}
        ts_path_pairs = []
        for ts in all_ts:
            stem = str(int(round(ts * 1e9)))
            if stem in img_map:
                ts_path_pairs.append((ts, img_map[stem]))
        ts_path_pairs.sort(key=lambda x: x[0])

        total = len(ts_path_pairs)
        if total == 0:
            raise RuntimeError(
                f"No images found in {self.cam0_dir}. "
                f"Re-run download() to fetch image files."
            )

        sel_idx  = self._select_frame_indices(total)
        sel      = [ts_path_pairs[i] for i in sel_idx]
        image_timestamps = [p[0] for p in sel]
        image_paths      = [p[1] for p in sel]
        N = len(image_paths)
        print(f"[TUM-VI] Selected {N}/{total} frames from '{self.sequence}'")

        # ---- images ---------------------------------------------------
        to_tensor = TF.ToTensor()
        tensors = [to_tensor(PILImage.open(p).convert("RGB")) for p in image_paths]
        images_np = np.stack([t.numpy() for t in tensors])
        print(f"[TUM-VI] Images loaded: {images_np.shape}")

        # ---- IMU -------------------------------------------------------
        imu_readings = parse_imu_csv(str(self.imu_csv))
        print(f"[TUM-VI] IMU readings : {len(imu_readings)}")

        # ---- Ground truth ---------------------------------------------
        gt_ts, gt_poses = parse_groundtruth_csv(str(self.mocap_csv))
        gt_extrinsics   = interpolate_groundtruth(gt_ts, gt_poses,
                                                   image_timestamps)
        print(f"[TUM-VI] GT loaded    : {len(gt_ts)} poses")

        prefix  = ("room" if self.sequence.startswith("room") else
                   "corridor" if self.sequence.startswith("corridor") else
                   "slides")
        cam_cal = _DEFAULT_CALIB.get(prefix, _DEFAULT_CALIB["room"])
        calib   = IMUCalibration(R_cam_imu=_R_CAM_IMU)

        return dict(
            images_np        = images_np,
            image_paths      = image_paths,
            image_timestamps = image_timestamps,
            imu_readings     = imu_readings,
            gt_extrinsics    = gt_extrinsics,
            gt_positions     = gt_poses[:, :3],
            gt_ts_all        = gt_ts,
            calib            = calib,
            cam_intrinsics   = cam_cal,
        )

    def __repr__(self) -> str:
        return (f"TUMVIDataset(sequence={self.sequence!r}, "
                f"n_frames={self.n_frames}, root={self._root})")
