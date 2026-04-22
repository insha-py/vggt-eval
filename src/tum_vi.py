"""
TUM Visual-Inertial Dataset loader for VGGT evaluation.

Dataset page: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
Format: EuRoC MAV

Download strategy
-----------------
The full tar archives are 1–2 GB each (cam0 + cam1 images + event data).
We avoid downloading the whole thing with a two-chunk approach:

  1. Fetch first 16 MB  → contains first ~60 cam0 images + cam0/data.csv
  2. Fetch last  16 MB  → contains imu0/data.csv + mocap0/data.csv
  3. Scan both chunks for POSIX tar headers (magic "ustar" + checksum)
  4. Targeted Range fetches for the ~40 images and 3 CSVs we actually need

Total transferred: ~30–45 MB instead of 1.6 GB.
Round-trips: ~45 (2 bulk + 43 individual) instead of 3000+.
Falls back to streaming tar if the server doesn't support Accept-Ranges.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.imu import (
    IMUCalibration,
    parse_imu_csv,
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

_REQUIRED_CSVS = {"cam0/data.csv", "imu0/data.csv", "mocap0/data.csv"}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return dict(r.headers)


def _http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ---------------------------------------------------------------------------
# Tar header parsing (POSIX ustar format)
# ---------------------------------------------------------------------------

_USTAR_MAGIC = b"ustar"


def _valid_tar_header(block: bytes) -> bool:
    """Return True only if this 512-byte block is a valid POSIX tar header."""
    if len(block) < 512:
        return False
    if block[257:262] != _USTAR_MAGIC:
        return False
    # Verify checksum: sum of all bytes with checksum field treated as spaces
    raw = block[148:156].strip(b"\x00 ")
    if not raw:
        return False
    try:
        stored = int(raw, 8)
    except ValueError:
        return False
    computed = sum(block[:148]) + sum(b"        ") + sum(block[156:])
    return computed == stored


def _parse_tar_header(block: bytes) -> Optional[Tuple[str, int, str]]:
    """
    Parse a valid 512-byte POSIX tar header.
    Returns (name, size_bytes, typeflag) or None.
    """
    try:
        name   = block[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        prefix = block[345:500].rstrip(b"\x00").decode("utf-8", errors="replace")
        if prefix:
            name = prefix.rstrip("/") + "/" + name
        size_f   = block[124:136].rstrip(b"\x00")
        size     = int(size_f or b"0", 8) if size_f else 0
        typeflag = chr(block[156]) if block[156] else "0"
    except Exception:
        return None
    return name, size, typeflag


def _scan_chunk(chunk: bytes, base_offset: int) -> Dict[str, Tuple[int, int]]:
    """
    Scan a raw bytes chunk at every 512-byte boundary for valid tar headers.
    Returns {member_name: (data_offset_in_archive, size_bytes)}.

    Uses POSIX magic ("ustar") + checksum validation to reject false positives
    from file data that happens to land on a 512-byte boundary.
    """
    members: Dict[str, Tuple[int, int]] = {}
    for i in range(0, len(chunk) - 512, 512):
        block = chunk[i : i + 512]
        if not _valid_tar_header(block):
            continue
        result = _parse_tar_header(block)
        if result is None:
            continue
        name, size, typeflag = result
        if typeflag in ("0", "\x00", ""):          # regular file
            data_offset = base_offset + i + 512
            members[name] = (data_offset, size)
    return members


# ---------------------------------------------------------------------------
# TUMVIDataset
# ---------------------------------------------------------------------------

class TUMVIDataset:
    """
    Downloads and loads a TUM VI sequence for VGGT + IMU evaluation.

    Uses two bulk HTTP Range fetches (first 16 MB + last 16 MB) to locate
    file entries, then fetches only the ~43 files we actually need.
    Typical download: ~35 MB in ~30 s instead of 1.6 GB.

    Args:
        sequence     : one of AVAILABLE_SEQUENCES, e.g. "room1"
        n_frames     : number of frames to evaluate (≤ 50 recommended)
        download_dir : where to store extracted files on disk
    """

    # Size of the two bulk fetches used to locate tar entries
    _SCAN_CHUNK_MB = 16

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
        self._root     = (self.download_dir
                          / f"dataset-{sequence}_512_16" / "mav0")

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

    def _csvs_ok(self) -> bool:
        return (self.cam0_csv.exists()
                and self.imu_csv.exists()
                and self.mocap_csv.exists())

    def is_downloaded(self) -> bool:
        return self._csvs_ok() and any(self.cam0_dir.glob("*.png"))

    # ------------------------------------------------------------------
    def download(self, force: bool = False) -> None:
        """
        Download only the files we need.
        Uses two-chunk Range scan then targeted fetches.
        Falls back to streaming if the server doesn't support Range.
        """
        if self.is_downloaded() and not force:
            n = len(list(self.cam0_dir.glob("*.png")))
            print(f"[TUM-VI] '{self.sequence}' already present "
                  f"({n} images, CSVs ✓)")
            return

        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Check Range support
        try:
            hdrs       = _http_head(self._url)
            total_size = int(hdrs.get("Content-Length", 0))
            can_range  = hdrs.get("Accept-Ranges", "").lower() == "bytes"
        except Exception as e:
            print(f"[TUM-VI] HEAD failed ({e}); streaming fallback.")
            can_range  = False
            total_size = 0

        if can_range and total_size > 0:
            ok = self._download_two_chunk(total_size)
            if ok:
                return
            print("[TUM-VI] Two-chunk method failed; streaming fallback …")

        self._download_streaming()

    # ------------------------------------------------------------------
    def _download_two_chunk(self, total_size: int) -> bool:
        """
        Scan the archive in up to 5 evenly-spaced 16 MB windows, then fetch
        only the files we need.

        Why multiple windows: the TUM VI room archives are ordered as
        cam0 (~500 MB) → cam1 (~500 MB) → imu0/mocap0 (~4 MB) → events0
        (~600 MB).  The CSVs sit at ~1000 MB, which is neither the start
        nor the end, so two-window (start + end) misses them.
        """
        chunk_b = self._SCAN_CHUNK_MB * 1024 * 1024

        # Evenly space 5 scan windows across the archive.
        # Round each start down to the nearest 512-byte boundary.
        n_windows  = 5
        offsets    = [
            int(total_size * i / (n_windows - 1) / 512) * 512
            for i in range(n_windows)
        ]
        # Clamp last window so it doesn't overshoot
        offsets[-1] = max(0, total_size - chunk_b)

        print(f"[TUM-VI] Scanning {total_size / 1024**2:.0f} MB archive "
              f"with {n_windows} × {self._SCAN_CHUNK_MB} MB windows …")

        members: Dict[str, Tuple[int, int]] = {}
        for idx, start in enumerate(offsets):
            end = min(start + chunk_b, total_size) - 1
            if end <= start:
                continue
            print(f"[TUM-VI]   Window {idx+1}/{n_windows}  "
                  f"[{start//1024**2}–{end//1024**2} MB] …",
                  end=" ", flush=True)
            chunk = _http_range(self._url, start, end)
            found = _scan_chunk(chunk, start)
            members.update(found)
            print(f"{len(found)} entries")

        print(f"[TUM-VI]   Found {len(members)} file entries in scanned chunks.")

        # ---- Identify what we need ------------------------------------
        images = sorted(
            [(name, off, sz) for name, (off, sz) in members.items()
             if "cam0/data/" in name and name.endswith(".png")],
            key=lambda x: x[0],
        )[: self.n_frames]

        csvs = [
            (name, off, sz) for name, (off, sz) in members.items()
            if self._output_path(name) in (self.cam0_csv, self.imu_csv, self.mocap_csv)
        ]

        if not images:
            print("[TUM-VI]   No cam0 images found in first chunk — "
                  "try increasing _SCAN_CHUNK_MB.")
            return False
        if len(csvs) < 3:
            missing = {self.cam0_csv, self.imu_csv, self.mocap_csv} - {
                self._output_path(n) for n, *_ in csvs
            }
            print(f"[TUM-VI]   Only {len(csvs)}/3 CSVs found. Missing: {missing}")
            return False

        # ---- Download each file with canonical output path -----------
        needed      = csvs + images
        total_bytes = sum(sz for _, _, sz in needed)
        print(f"[TUM-VI]   Downloading {len(needed)} files "
              f"({total_bytes / 1024**2:.1f} MB) …")

        for i, (name, offset, size) in enumerate(needed):
            out = self._output_path(name)
            if out is None:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)

            if out.exists() and out.stat().st_size == size:
                continue

            data = _http_range(self._url, offset, offset + size - 1)
            out.write_bytes(data)

            mb_so_far = sum(needed[j][2] for j in range(i + 1)) / 1024**2
            print(f"\r[TUM-VI]   {i+1}/{len(needed)}  ({mb_so_far:.1f} MB)",
                  end="", flush=True)

        print(f"\n[TUM-VI] Done — {len(images)} images + {len(csvs)} CSVs saved.")
        return True

    # ------------------------------------------------------------------
    def _download_streaming(self) -> None:
        """
        Fallback: stream the tar, extract first N images + CSVs, then stop.
        """
        print(f"[TUM-VI] Streaming {self._url} …")
        images_saved = 0
        csvs_saved: set = set()

        try:
            with urllib.request.urlopen(self._url, timeout=180) as resp:
                with tarfile.open(fileobj=resp, mode="r|") as tf:
                    for member in tf:
                        out = self._output_path(member.name)
                        if out is None:
                            continue
                        is_img = (out.parent == self.cam0_dir
                                  and images_saved < self.n_frames)
                        is_csv = out in (self.cam0_csv, self.imu_csv, self.mocap_csv)
                        if not (is_img or is_csv):
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        fobj = tf.extractfile(member)
                        if fobj:
                            out.write_bytes(fobj.read())
                        if is_img:
                            images_saved += 1
                        if is_csv:
                            csvs_saved.add(str(out))
                        print(f"\r[TUM-VI]   images={images_saved}/{self.n_frames}"
                              f"  csvs={len(csvs_saved)}/3",
                              end="", flush=True)
                        if images_saved >= self.n_frames and len(csvs_saved) >= 3:
                            print("\n[TUM-VI] Got everything; closing stream.")
                            break
        except Exception as e:
            if images_saved > 0 and len(csvs_saved) == 3:
                print(f"\n[TUM-VI] Stream done ({e}).")
            elif images_saved == 0:
                raise

    # ------------------------------------------------------------------
    def _output_path(self, tar_name: str) -> Optional[Path]:
        """
        Map a tar member name to its fixed on-disk output path.

        This is intentionally independent of how many prefix components the
        tar uses (dataset-room1_512_16/mav0/... vs mav0/... vs plain paths).
        We match on the *suffix* of the member name and return the canonical
        path relative to self._root.

        Returns None if this member is not one we want to save.
        """
        n = tar_name.replace("\\", "/").lstrip("./")
        if n.endswith("cam0/data.csv"):
            return self.cam0_csv
        if n.endswith("imu0/data.csv"):
            return self.imu_csv
        if n.endswith("mocap0/data.csv"):
            return self.mocap_csv
        if "cam0/data/" in n and n.endswith(".png"):
            return self.cam0_dir / Path(n).name
        return None

    # ------------------------------------------------------------------
    def _select_frame_indices(self, total: int) -> List[int]:
        return np.linspace(0, total - 1,
                           min(self.n_frames, total), dtype=int).tolist()

    # ------------------------------------------------------------------
    def load(self) -> dict:
        """
        Load the dataset into memory.

        Returns dict with keys:
            images_np, image_paths, image_timestamps,
            imu_readings, gt_extrinsics, gt_positions,
            gt_ts_all, calib, cam_intrinsics
        """
        if not self.is_downloaded():
            raise RuntimeError(
                f"Dataset not ready at {self._root}. "
                "Call download() first."
            )

        from PIL import Image as PILImage
        from torchvision import transforms as TF

        # ---- timestamps + paths --------------------------------------
        # Derive timestamps directly from PNG filenames (stem = ns timestamp).
        # Avoids float64 round-trip loss: TUM VI stems are ~1.5e18 ns (19
        # significant digits), which float64 cannot represent exactly.
        all_paths = sorted(self.cam0_dir.glob("*.png"), key=lambda p: p.name)
        ts_path_pairs = []
        for p in all_paths:
            try:
                ts_ns = int(p.stem)           # exact integer, no precision loss
                ts_path_pairs.append((ts_ns * 1e-9, str(p)))
            except ValueError:
                pass

        total = len(ts_path_pairs)
        if total == 0:
            raise RuntimeError(
                f"No images found in {self.cam0_dir}. "
                "Re-run download() to fetch images."
            )

        sel          = [ts_path_pairs[i] for i in self._select_frame_indices(total)]
        image_timestamps = [p[0] for p in sel]
        image_paths      = [p[1] for p in sel]
        N = len(image_paths)
        print(f"[TUM-VI] Selected {N}/{total} frames from '{self.sequence}'")

        # ---- images --------------------------------------------------
        to_tensor = TF.ToTensor()
        images_np = np.stack([
            to_tensor(PILImage.open(p).convert("RGB")).numpy()
            for p in image_paths
        ])
        print(f"[TUM-VI] Images: {images_np.shape}")

        # ---- IMU -----------------------------------------------------
        imu_readings = parse_imu_csv(str(self.imu_csv))
        print(f"[TUM-VI] IMU readings: {len(imu_readings)}")

        # ---- Ground truth --------------------------------------------
        gt_ts, gt_poses = parse_groundtruth_csv(str(self.mocap_csv))
        gt_extrinsics   = interpolate_groundtruth(gt_ts, gt_poses,
                                                   image_timestamps)
        print(f"[TUM-VI] GT: {len(gt_ts)} poses")

        prefix  = ("room"     if self.sequence.startswith("room")     else
                   "corridor" if self.sequence.startswith("corridor") else
                   "slides")
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
            cam_intrinsics   = _DEFAULT_CALIB.get(prefix, _DEFAULT_CALIB["room"]),
        )

    def __repr__(self) -> str:
        return (f"TUMVIDataset(sequence={self.sequence!r}, "
                f"n_frames={self.n_frames}, root={self._root})")
