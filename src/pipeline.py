"""
Core VGGT inference pipeline (Phase 2).

Usage
-----
from src.pipeline import VGGTPipeline

pipe = VGGTPipeline()
pipe.load_model()

result = pipe.run(image_dir="path/to/images/")
# result keys: extrinsic, intrinsic, depth_map, depth_conf, point_map, images

pipe.save_ply(result, "results/points.ply")
pipe.save_colmap(result, image_dir, "results/sparse/")
"""

import os
import glob
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------
# VGGT model URL
# -----------------------------------------------------------------
_VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_VGGT_RESOLUTION = 518   # model's native input resolution


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _get_device_dtype() -> Tuple[str, torch.dtype]:
    """Choose device and bfloat16 / float16 based on hardware capability."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


def _list_images(directory: str) -> List[str]:
    """Return sorted list of image paths from a directory."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))


# -----------------------------------------------------------------
# Image loading  (mirrors VGGT's own load_fn but self-contained)
# -----------------------------------------------------------------

def load_images_from_dir(
    image_dir: str,
    target_size: int = _VGGT_RESOLUTION,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """
    Load, square-pad, and resize images from a directory.

    Returns
    -------
    images        : (N, 3, target_size, target_size)  float32 in [0,1]
    orig_coords   : (N, 6)  [x1, y1, x2, y2, W, H] in target-pixel space
    image_paths   : list of absolute file paths (sorted)
    """
    from PIL import Image as PILImage
    from torchvision import transforms as TF

    paths = _list_images(image_dir)
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    to_tensor = TF.ToTensor()
    tensors   = []
    coords    = []

    for p in paths:
        img = PILImage.open(p)
        if img.mode == "RGBA":
            bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
            img = PILImage.alpha_composite(bg, img)
        img = img.convert("RGB")
        W, H = img.size

        max_dim = max(W, H)
        left = (max_dim - W) // 2
        top  = (max_dim - H) // 2
        scale = target_size / max_dim
        x1, y1 = left * scale, top * scale
        x2, y2 = (left + W) * scale, (top + H) * scale
        coords.append(np.array([x1, y1, x2, y2, W, H], dtype=np.float32))

        sq = PILImage.new("RGB", (max_dim, max_dim), (0, 0, 0))
        sq.paste(img, (left, top))
        sq = sq.resize((target_size, target_size), PILImage.Resampling.BICUBIC)
        tensors.append(to_tensor(sq))

    images = torch.stack(tensors)                     # (N, 3, H, W)
    orig_coords = np.stack(coords)                    # (N, 6)
    return images, orig_coords, paths


def load_images_from_list(
    image_paths: List[str],
    target_size: int = _VGGT_RESOLUTION,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Like load_images_from_dir but from an explicit list.
    Returns (images, orig_coords) without the path list.
    """
    from PIL import Image as PILImage
    from torchvision import transforms as TF

    to_tensor = TF.ToTensor()
    tensors, coords = [], []

    for p in image_paths:
        img = PILImage.open(p)
        if img.mode == "RGBA":
            bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
            img = PILImage.alpha_composite(bg, img)
        img = img.convert("RGB")
        W, H = img.size
        max_dim = max(W, H)
        left = (max_dim - W) // 2
        top  = (max_dim - H) // 2
        scale = target_size / max_dim
        coords.append(np.array(
            [left * scale, top * scale,
             (left + W) * scale, (top + H) * scale, W, H],
            dtype=np.float32,
        ))
        sq = PILImage.new("RGB", (max_dim, max_dim), (0, 0, 0))
        sq.paste(img, (left, top))
        sq = sq.resize((target_size, target_size), PILImage.Resampling.BICUBIC)
        tensors.append(to_tensor(sq))

    return torch.stack(tensors), np.stack(coords)


# -----------------------------------------------------------------
# Core inference call
# -----------------------------------------------------------------

def run_vggt_inference(
    model,
    images: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    resolution: int = _VGGT_RESOLUTION,
) -> Dict[str, np.ndarray]:
    """
    Run VGGT on a batch of images.

    Args:
        model  : loaded VGGT model (already on device)
        images : (N, 3, H, W) float32 in [0,1]

    Returns dict with keys:
        extrinsic  : (N, 3, 4) camera-from-world   (OpenCV convention)
        intrinsic  : (N, 3, 3) camera intrinsics
        depth_map  : (N, H, W, 1) metric depth
        depth_conf : (N, H, W)   confidence in [0,1]
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    assert images.ndim == 4 and images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution),
                           mode="bilinear", align_corners=False)
    images = images.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            imgs_batched = images[None]             # (1, N, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(imgs_batched)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs_batched.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, imgs_batched, ps_idx)

    return {
        "extrinsic":  extrinsic.squeeze(0).cpu().numpy(),
        "intrinsic":  intrinsic.squeeze(0).cpu().numpy(),
        "depth_map":  depth_map.squeeze(0).cpu().numpy(),
        "depth_conf": depth_conf.squeeze(0).cpu().numpy(),
    }


# -----------------------------------------------------------------
# Point cloud from depth maps
# -----------------------------------------------------------------

def depth_to_point_cloud(
    depth_map:   np.ndarray,
    depth_conf:  np.ndarray,
    extrinsic:   np.ndarray,
    intrinsic:   np.ndarray,
    images_rgb:  Optional[np.ndarray] = None,
    conf_thresh: float = 5.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unproject depth maps to a 3-D point cloud in world coordinates.

    Args:
        depth_map   : (N, H, W, 1) or (N, H, W)
        depth_conf  : (N, H, W)  log-scale confidence from VGGT
        extrinsic   : (N, 3, 4)
        intrinsic   : (N, 3, 3)
        images_rgb  : (N, 3, H, W) float32 in [0,1]  for per-point colour
        conf_thresh : points with conf < thresh are discarded

    Returns:
        points (M, 3), colors (M, 3) uint8 or None
    """
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    depth = np.squeeze(depth_map)    # (N, H, W)
    conf  = np.squeeze(depth_conf)   # (N, H, W)

    # Unproject: (N, H, W, 3)
    points_3d = unproject_depth_map_to_point_map(depth[:, :, :, None],
                                                  extrinsic, intrinsic)

    N, H, W, _ = points_3d.shape
    points_flat = points_3d.reshape(-1, 3)
    conf_flat   = conf.reshape(-1)

    mask = conf_flat >= conf_thresh

    colors_flat = None
    if images_rgb is not None:
        # images_rgb: (N,3,H,W) -> (N,H,W,3) -> (N*H*W, 3)
        rgb = np.transpose(images_rgb, (0, 2, 3, 1))  # (N,H,W,3)
        rgb_flat = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
        colors_flat = rgb_flat[mask]

    return points_flat[mask], colors_flat


# -----------------------------------------------------------------
# VGGTPipeline  — main class
# -----------------------------------------------------------------

class VGGTPipeline:
    """
    End-to-end VGGT inference pipeline.

    Example
    -------
    pipe = VGGTPipeline()
    pipe.load_model()
    result = pipe.run("data/scene/images/")
    pipe.save_ply(result, "out/points.ply")
    """

    def __init__(self, model_url: str = _VGGT_MODEL_URL):
        self.model_url  = model_url
        self.model      = None
        self.device, self.dtype = _get_device_dtype()
        print(f"[Pipeline] device={self.device}  dtype={self.dtype}")

    # ------------------------------------------------------------------
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load VGGT-1B.  If checkpoint_path is provided and exists, load from
        disk; otherwise download from HuggingFace.
        """
        from vggt.models.vggt import VGGT

        print("[Pipeline] Loading VGGT model …")
        t0 = time.time()
        self.model = VGGT()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"[Pipeline] Loaded from {checkpoint_path}")
        else:
            state = torch.hub.load_state_dict_from_url(self.model_url, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"[Pipeline] Downloaded from HuggingFace")

        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"[Pipeline] Model ready ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    def run(
        self,
        image_dir: str,
        max_frames: Optional[int] = None,
        conf_thresh: float = 5.0,
    ) -> Dict:
        """
        Full inference pipeline on a directory of images.

        Returns
        -------
        dict with:
            extrinsic  (N,3,4), intrinsic (N,3,3),
            depth_map  (N,H,W,1), depth_conf (N,H,W),
            point_cloud (M,3) float32,  point_colors (M,3) uint8,
            images_np   (N,3,H,W) float32,
            image_paths list[str],
            orig_coords (N,6),
            inference_time_s  float,
            peak_gpu_mb       float,
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        print(f"[Pipeline] Loading images from {image_dir}")
        images, orig_coords, paths = load_images_from_dir(image_dir, max_frames=max_frames)
        N = len(images)
        print(f"[Pipeline] {N} frames loaded")

        # --- memory & timing ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        raw = run_vggt_inference(self.model, images, self.device, self.dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if torch.cuda.is_available() else 0.0)

        print(f"[Pipeline] Inference done in {elapsed:.2f}s  "
              f"peak GPU {peak_mb:.0f} MB")

        # --- point cloud ---
        pts, clrs = depth_to_point_cloud(
            raw["depth_map"], raw["depth_conf"],
            raw["extrinsic"], raw["intrinsic"],
            images_rgb=images.numpy(),
            conf_thresh=conf_thresh,
        )
        print(f"[Pipeline] Point cloud: {len(pts):,} points")

        result = dict(
            **raw,
            point_cloud     = pts,
            point_colors    = clrs,
            images_np       = images.numpy(),
            image_paths     = paths,
            orig_coords     = orig_coords,
            inference_time_s = elapsed,
            peak_gpu_mb     = peak_mb,
        )
        return result

    # ------------------------------------------------------------------
    def save_ply(
        self,
        result: Dict,
        output_path: str,
    ) -> None:
        """Save the point cloud from a pipeline result to a PLY file."""
        from src.visualization import save_ply
        save_ply(output_path, result["point_cloud"], result.get("point_colors"))
        print(f"[Pipeline] PLY saved to {output_path}")

    # ------------------------------------------------------------------
    def save_colmap(
        self,
        result: Dict,
        image_dir: str,
        output_dir: str,
        shared_camera: bool = False,
        camera_type:   str  = "SIMPLE_PINHOLE",
        conf_thresh:   float = 5.0,
    ) -> None:
        """
        Export reconstruction to COLMAP sparse format.
        Requires pycolmap and trimesh to be installed.
        """
        try:
            import pycolmap
            import trimesh
            from vggt.dependency.np_to_pycolmap import (
                batch_np_matrix_to_pycolmap_wo_track,
            )
        except ImportError as e:
            print(f"[Pipeline] Cannot save COLMAP: {e}")
            return

        import copy
        from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        os.makedirs(output_dir, exist_ok=True)

        extrinsic   = result["extrinsic"]
        intrinsic   = result["intrinsic"]
        depth_map   = result["depth_map"]
        depth_conf  = result["depth_conf"]
        orig_coords = result["orig_coords"]
        paths       = result["image_paths"]
        N           = len(extrinsic)

        depth_sq = np.squeeze(depth_map)
        conf_sq  = np.squeeze(depth_conf)
        H, W     = depth_sq.shape[1], depth_sq.shape[2]
        image_size = (H, W)

        points_3d_all = unproject_depth_map_to_point_map(
            depth_sq[:, :, :, None], extrinsic, intrinsic
        )

        pixel_coords = create_pixel_coordinate_grid(N, H, W)

        MAX_COLMAP = 100_000
        conf_mask = conf_sq.reshape(-1) >= conf_thresh
        conf_mask = randomly_limit_trues(conf_mask, MAX_COLMAP)

        pts3d  = points_3d_all.reshape(-1, 3)[conf_mask]
        xyf    = pixel_coords.reshape(-1, 3)[conf_mask]
        rgb    = (result["images_np"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)[conf_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            pts3d, xyf, rgb, extrinsic, intrinsic, image_size,
            shared_camera=shared_camera, camera_type=camera_type,
        )

        # Rename images and rescale cameras to original resolution
        for pid in reconstruction.images:
            pyimg = reconstruction.images[pid]
            pycam = reconstruction.cameras[pyimg.camera_id]
            pyimg.name = os.path.basename(paths[pid - 1])

            real_wh = orig_coords[pid - 1, 4:6]          # original W, H
            ratio   = max(real_wh) / _VGGT_RESOLUTION
            params  = copy.deepcopy(pycam.params)
            params  = params * ratio
            params[-2:] = real_wh / 2                      # principal point at centre
            pycam.params = params
            pycam.width  = int(real_wh[0])
            pycam.height = int(real_wh[1])

        reconstruction.write(output_dir)
        ply_path = os.path.join(output_dir, "points.ply")
        trimesh.PointCloud(pts3d, colors=rgb).export(ply_path)
        print(f"[Pipeline] COLMAP sparse saved to {output_dir}")


# -----------------------------------------------------------------
# Convenience: print a summary of a result dict
# -----------------------------------------------------------------

def print_result_summary(result: Dict) -> None:
    N = len(result["extrinsic"])
    M = len(result["point_cloud"])
    print(f"\n{'─'*40}")
    print(f"  VGGT Result Summary")
    print(f"{'─'*40}")
    print(f"  Frames         : {N}")
    print(f"  Point cloud    : {M:,} points")
    print(f"  Inference time : {result['inference_time_s']:.2f} s")
    print(f"  Peak GPU mem   : {result['peak_gpu_mb']:.0f} MB "
          f"({result['peak_gpu_mb']/1024:.2f} GB)")
    print(f"  Extrinsic shape: {result['extrinsic'].shape}")
    print(f"  Intrinsic shape: {result['intrinsic'].shape}")
    print(f"  Depth map shape: {result['depth_map'].shape}")
    print(f"{'─'*40}\n")
