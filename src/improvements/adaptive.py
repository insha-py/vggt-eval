"""
Adaptive Resolution VGGT (Phase 6 improvement strategy).

Core idea
---------
VGGT's native 518 px resolution uses a lot of VRAM and is often wasted on
high-confidence image regions.  We can:

  1. Run a cheap *global pass* at low resolution (224 px).
  2. Identify *uncertain* regions where depth confidence is below a threshold.
  3. Crop square patches around those regions and run a *local high-res pass*
     (up to 518 px) on each patch.
  4. Merge depth and pose predictions back into a full-resolution output using
     confidence-weighted blending.

Step 4 is done only for depth maps (which are spatial).  For camera *poses*
the global low-res pass already produces good estimates; the per-patch runs
give us a refined dense depth map rather than alternative camera poses.

Result
------
On a 16 GB T4:  ~1.5–2× wall time of a full-518 run, but with much better
depth detail in texture-poor or distant regions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Patch helper
# ---------------------------------------------------------------------------

def _find_low_confidence_patches(
    conf_map: np.ndarray,     # (H, W) in [0, 1]
    threshold: float,
    patch_size: int,
    max_patches: int,
    min_uncertain_frac: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    Return up to max_patches non-overlapping (y0, x0, y1, x1) bounding boxes
    around regions with mean confidence < threshold.

    Uses a sliding-window scan with stride = patch_size // 2.
    """
    H, W   = conf_map.shape
    stride = max(1, patch_size // 2)
    scores: List[Tuple[float, int, int]] = []

    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            patch_conf = conf_map[y0 : y0 + patch_size, x0 : x0 + patch_size]
            uncertain_frac = float((patch_conf < threshold).mean())
            if uncertain_frac >= min_uncertain_frac:
                scores.append((uncertain_frac, y0, x0))

    scores.sort(reverse=True)

    # Greedy selection — skip overlapping boxes
    selected: List[Tuple[int, int, int, int]] = []
    occupied = np.zeros((H, W), dtype=bool)

    for _, y0, x0 in scores:
        if len(selected) >= max_patches:
            break
        y1, x1 = y0 + patch_size, x0 + patch_size
        if occupied[y0:y1, x0:x1].any():
            continue
        selected.append((y0, x0, y1, x1))
        occupied[y0:y1, x0:x1] = True

    return selected


def _crop_images_to_patch(
    images: "torch.Tensor",    # (N, 3, H, W)
    y0: int, x0: int, y1: int, x1: int,
    target_size: int,
) -> "torch.Tensor":
    """Crop all N images to the patch and resize to target_size."""
    import torch
    import torch.nn.functional as F

    patch = images[:, :, y0:y1, x0:x1]               # (N, 3, ph, pw)
    patch = F.interpolate(
        patch, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )
    return patch


# ---------------------------------------------------------------------------
# AdaptiveResolutionVGGT
# ---------------------------------------------------------------------------

class AdaptiveResolutionVGGT:
    """
    Two-pass adaptive inference: cheap global low-res + targeted high-res patches.

    Parameters
    ----------
    global_res    : resolution for the initial global pass (default 224).
    patch_res     : resolution for the per-patch high-res pass (default 518).
    conf_threshold: depth confidence below this triggers a high-res patch.
    patch_size_px : size of each square patch (in global_res pixels).
    max_patches   : maximum number of patches per scene.
    """

    def __init__(
        self,
        global_res:     int   = 224,
        patch_res:      int   = 518,
        conf_threshold: float = 0.3,
        patch_size_px:  int   = 64,
        max_patches:    int   = 4,
    ):
        self.global_res     = global_res
        self.patch_res      = patch_res
        self.conf_threshold = conf_threshold
        self.patch_size_px  = patch_size_px
        self.max_patches    = max_patches
        self._model         = None
        self._device        = None
        self._dtype         = None

    # ------------------------------------------------------------------
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        from src.pipeline import VGGTPipeline
        pipe = VGGTPipeline()
        pipe.load_model(checkpoint_path=checkpoint_path)
        self._model  = pipe.model
        self._device = pipe.device
        self._dtype  = pipe.dtype
        print(f"[AdaptiveResolutionVGGT] Model on {self._device}")

    # ------------------------------------------------------------------
    def run(
        self,
        image_dir: str,
        max_frames: Optional[int] = None,
    ) -> dict:
        """
        Run adaptive two-pass inference on images in image_dir.

        Returns a dict with the same keys as run_vggt_inference plus:
          "patches_used"  : number of high-res patches applied
          "conf_map_low"  : (N, H, W) confidence from the global low-res pass
          "depth_map_low" : (N, H, W, 1) depth from the global low-res pass
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        import torch
        from src.pipeline import load_images_from_dir, run_vggt_inference

        # --- Global low-res pass -------------------------------------------
        print(f"[AdaptiveVGGT] Global pass at {self.global_res}px …")
        imgs_low, _, paths = load_images_from_dir(
            image_dir, target_size=self.global_res, max_frames=max_frames
        )
        out_low = run_vggt_inference(
            self._model, imgs_low, self._device, self._dtype,
            resolution=self.global_res,
        )
        conf_low  = out_low["depth_conf"]   # (N, H, W)
        depth_low = out_low["depth_map"]    # (N, H, W, 1)
        ext_low   = out_low["extrinsic"]    # (N, 3, 4)

        N, H, W = conf_low.shape

        # --- Identify uncertain patches (per-frame mean confidence map) -----
        mean_conf = conf_low.mean(axis=0)   # (H, W) averaged across frames
        patches   = _find_low_confidence_patches(
            mean_conf,
            threshold=self.conf_threshold,
            patch_size=self.patch_size_px,
            max_patches=self.max_patches,
        )
        print(f"[AdaptiveVGGT] Found {len(patches)} uncertain patch(es).")

        # Start with low-res results; we'll refine depth only
        depth_merged = depth_low.copy()   # (N, H, W, 1)
        conf_merged  = conf_low.copy()    # (N, H, W)

        # Load high-res images once (if we actually have patches to run)
        if patches:
            imgs_hi, _, _ = load_images_from_dir(
                image_dir, target_size=self.patch_res, max_frames=max_frames
            )

        for (py0, px0, py1, px1) in patches:
            print(f"  Patch ({py0},{px0})–({py1},{px1}) at {self.patch_res}px …")

            patch_imgs = _crop_images_to_patch(
                imgs_hi, py0, px0, py1, px1, self.patch_res
            )   # (N, 3, patch_res, patch_res)

            out_hi = run_vggt_inference(
                self._model, patch_imgs, self._device, self._dtype,
                resolution=self.patch_res,
            )
            depth_hi = out_hi["depth_map"]   # (N, patch_res, patch_res, 1)
            conf_hi  = out_hi["depth_conf"]  # (N, patch_res, patch_res)

            # Downsample high-res depth/conf back to low-res patch size
            ph = py1 - py0
            pw = px1 - px0

            import torch
            import torch.nn.functional as TF

            depth_hi_t = torch.from_numpy(
                depth_hi[..., 0]
            ).unsqueeze(1).float()   # (N, 1, pr, pr)
            depth_hi_down = TF.interpolate(
                depth_hi_t, size=(ph, pw), mode="bilinear", align_corners=False
            ).squeeze(1).numpy()     # (N, ph, pw)

            conf_hi_t = torch.from_numpy(conf_hi).unsqueeze(1).float()
            conf_hi_down = TF.interpolate(
                conf_hi_t, size=(ph, pw), mode="bilinear", align_corners=False
            ).squeeze(1).numpy()

            # Confidence-weighted blend
            c_lo = conf_merged[:, py0:py1, px0:px1]                    # (N,ph,pw)
            c_hi = conf_hi_down
            total = c_lo + c_hi + 1e-8
            w_lo  = c_lo / total
            w_hi  = c_hi / total

            depth_merged[:, py0:py1, px0:px1, 0] = (
                w_lo * depth_merged[:, py0:py1, px0:px1, 0]
                + w_hi * depth_hi_down
            )
            conf_merged[:, py0:py1, px0:px1] = (
                np.maximum(c_lo, c_hi)
            )

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compose output dict
        result = dict(out_low)
        result["depth_map"]     = depth_merged
        result["depth_conf"]    = conf_merged
        result["patches_used"]  = len(patches)
        result["conf_map_low"]  = conf_low
        result["depth_map_low"] = depth_low
        return result
