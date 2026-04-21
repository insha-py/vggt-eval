"""
3-D visualisation utilities for VGGT reconstruction results.

Functions:
  - plot_point_cloud          : interactive Plotly scatter-3D
  - plot_depth_maps           : grid of depth / confidence images (matplotlib)
  - plot_cameras              : camera frustum overlay on a point cloud
  - plot_trajectory           : predicted vs. GT camera trajectory
  - plot_memory_vs_frames     : benchmark chart
  - plot_chunk_alignment      : visualise per-chunk colour-coded reconstruction
  - save_ply                  : write point cloud to .ply file
  - load_ply                  : load a .ply file to numpy arrays
"""

import os
import struct
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def save_ply(
    path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> None:
    """
    Write a point cloud to a binary-little-endian PLY file.

    Args:
        path:    output file path
        points:  (N, 3) float32 XYZ
        colors:  (N, 3) uint8  RGB  (optional)
        normals: (N, 3) float32 normals (optional)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    N = len(points)

    props = ["x", "y", "z"]
    fmt   = "fff"
    rows  = [points]

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        props += ["nx", "ny", "nz"]
        fmt   += "fff"
        rows.append(normals)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        props += ["red", "green", "blue"]
        fmt   += "BBB"
        rows.append(colors)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
    )
    for p in props:
        dtype = "float" if p not in ("red", "green", "blue") else "uchar"
        header += f"property {dtype} {p}\n"
    header += "end_header\n"

    # Build a structured numpy dtype so each column keeps its own type
    dt_map = {"f": np.float32, "B": np.uint8}
    np_dt  = np.dtype([(p, dt_map[c]) for p, c in zip(props, fmt)])
    struct_arr = np.empty(N, dtype=np_dt)
    col = 0
    for arr in rows:
        for ci in range(arr.shape[1]):
            struct_arr[props[col]] = arr[:, ci]
            col += 1

    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(struct_arr.tobytes())


def load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load XYZ (and optionally RGB) from a PLY file.
    Falls back to open3d if available, otherwise uses a minimal parser.

    Returns:
        points (N,3) float32, colors (N,3) uint8 or None
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        clr = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        return pts, clr
    except ImportError:
        pass

    # Minimal ASCII / binary parser
    with open(path, "rb") as f:
        raw = f.read()

    lines  = raw.split(b"\n")
    n_vert = 0
    props  = []
    end_hdr = 0
    for i, line in enumerate(lines):
        s = line.decode(errors="replace").strip()
        if s.startswith("element vertex"):
            n_vert = int(s.split()[-1])
        elif s.startswith("property"):
            parts = s.split()
            props.append((parts[1], parts[2]))   # (type, name)
        elif s == "end_header":
            end_hdr = sum(len(l) + 1 for l in lines[: i + 1])
            break

    # Build numpy dtype
    type_map = {"float": "f4", "double": "f8", "uchar": "u1",
                "int": "i4", "uint": "u4", "short": "i2"}
    dt = np.dtype([(name, type_map.get(t, "f4")) for t, name in props])
    arr = np.frombuffer(raw[end_hdr:], dtype=dt, count=n_vert)

    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    clr = None
    if "red" in arr.dtype.names:
        clr = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)

    return pts, clr


# ---------------------------------------------------------------------------
# Plotly helpers (lazy import so matplotlib-only installs still work)
# ---------------------------------------------------------------------------

def _plotly_scatter3d(points, colors_rgb01, size=1.5, opacity=0.8, name=""):
    """Return a go.Scatter3d trace."""
    import plotly.graph_objects as go

    if colors_rgb01 is not None:
        color_str = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                     for r, g, b in colors_rgb01]
    else:
        color_str = points[:, 2]   # height-coloured

    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        name=name,
        marker=dict(size=size, color=color_str, opacity=opacity),
    )


def _blank_3d_layout():
    import plotly.graph_objects as go
    return go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(15,15,15)",
        ),
        paper_bgcolor="rgb(15,15,15)",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(font=dict(color="white")),
    )


# ---------------------------------------------------------------------------
# Point cloud visualisation
# ---------------------------------------------------------------------------

def plot_point_cloud(
    points:       np.ndarray,
    colors:       Optional[np.ndarray] = None,
    title:        str  = "Point Cloud",
    max_points:   int  = 200_000,
    point_size:   float = 1.5,
    save_html:    Optional[str] = None,
    show:         bool = True,
):
    """
    Interactive Plotly 3-D point cloud viewer.

    Args:
        points:     (N, 3) XYZ
        colors:     (N, 3) float RGB in [0,1] or uint8 in [0,255]  (optional)
        title:      plot title
        max_points: subsample above this number
        save_html:  path to save interactive HTML (optional)
        show:       call fig.show()
    """
    import plotly.graph_objects as go

    if len(points) > max_points:
        idx    = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    if colors is not None:
        colors = np.asarray(colors, dtype=float)
        if colors.max() > 1.0:
            colors = colors / 255.0

    layout = _blank_3d_layout()
    layout.title = dict(text=title, font=dict(color="white"))

    fig = go.Figure(
        data=[_plotly_scatter3d(points, colors)],
        layout=layout,
    )

    if save_html:
        os.makedirs(os.path.dirname(os.path.abspath(save_html)), exist_ok=True)
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Depth map grid
# ---------------------------------------------------------------------------

def plot_depth_maps(
    depth_maps:   np.ndarray,
    conf_maps:    Optional[np.ndarray] = None,
    n_cols:       int  = 4,
    figsize_per:  Tuple[float, float] = (3.0, 3.0),
    title:        str  = "Depth Maps",
    save_path:    Optional[str] = None,
    show:         bool = True,
):
    """
    Display a grid of depth maps (and optionally confidence maps).

    Args:
        depth_maps: (N, H, W) or (N, H, W, 1) depth in metres
        conf_maps:  (N, H, W) confidence in [0,1]  (optional)
        n_cols:     columns per row
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    depth_maps = np.squeeze(depth_maps)       # -> (N, H, W)
    N     = len(depth_maps)
    ncols = min(n_cols, N)
    nrows = int(np.ceil(N / ncols)) * (2 if conf_maps is not None else 1)
    fw, fh = figsize_per
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw * ncols, fh * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i in range(N):
        row = (i // ncols) * (2 if conf_maps is not None else 1)
        col = i % ncols
        ax  = axes[row, col]
        d   = depth_maps[i]
        vmin, vmax = np.percentile(d, [2, 98])
        im = ax.imshow(d, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(f"frame {i}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if conf_maps is not None:
            ax2 = axes[row + 1, col]
            ax2.imshow(conf_maps[i], cmap="viridis", vmin=0, vmax=1)
            ax2.set_title(f"conf {i}", fontsize=8)
            ax2.axis("off")

    # Hide unused axes
    for i in range(N, nrows // (2 if conf_maps is not None else 1) * ncols):
        row = (i // ncols) * (2 if conf_maps is not None else 1)
        col = i % ncols
        axes[row, col].axis("off")
        if conf_maps is not None and row + 1 < nrows:
            axes[row + 1, col].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Camera frustum visualisation
# ---------------------------------------------------------------------------

def _frustum_lines(extrinsic: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Return line endpoints (5 edges × 2 points, shape (10, 3)) for a camera frustum.
    extrinsic: (3, 4) camera-from-world.
    """
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    c = -R.T @ t  # camera centre in world

    # Four corners of the image plane in camera frame (normalised)
    corners_cam = np.array([
        [ 1,  1, 1],
        [-1,  1, 1],
        [-1, -1, 1],
        [ 1, -1, 1],
    ], dtype=float) * scale

    corners_world = (R.T @ (corners_cam - t).T).T  # wrong if t is already in camera frame
    # Correct unprojection: p_world = R^T (p_cam - t)
    corners_world = (R.T @ corners_cam.T).T + c

    lines = []
    for corner in corners_world:
        lines.append(c)
        lines.append(corner)
    # Draw box edge: connect corners in order
    for i in range(4):
        lines.append(corners_world[i])
        lines.append(corners_world[(i + 1) % 4])
    return np.array(lines)


def plot_cameras(
    extrinsics:   np.ndarray,
    points:       Optional[np.ndarray] = None,
    point_colors: Optional[np.ndarray] = None,
    gt_extrinsics: Optional[np.ndarray] = None,
    frustum_scale: float = 0.05,
    title:        str = "Camera Poses",
    max_points:   int = 100_000,
    save_html:    Optional[str] = None,
    show:         bool = True,
):
    """
    3-D camera frustum visualisation (Plotly).

    Blue = predicted cameras, green = GT cameras (if provided).
    """
    import plotly.graph_objects as go

    traces = []

    def _add_camera_traces(extrs, color, name):
        centers = []
        for E in extrs:
            R, t = E[:3, :3], E[:3, 3]
            centers.append(-R.T @ t)

        centers = np.array(centers)
        traces.append(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers+lines",
            name=name + " centres",
            marker=dict(size=4, color=color),
            line=dict(color=color, width=2),
        ))
        # Frustum edges
        all_pts: List[np.ndarray] = []
        for E in extrs:
            lines = _frustum_lines(E, scale=frustum_scale)
            all_pts.append(lines)
            all_pts.append(np.array([[None, None, None]]))
        edge_pts = np.concatenate(all_pts)
        traces.append(go.Scatter3d(
            x=edge_pts[:, 0], y=edge_pts[:, 1], z=edge_pts[:, 2],
            mode="lines",
            name=name + " frustums",
            line=dict(color=color, width=1),
            showlegend=False,
        ))

    _add_camera_traces(extrinsics, "#3399ff", "pred")

    if gt_extrinsics is not None:
        _add_camera_traces(gt_extrinsics, "#33cc66", "gt")

    if points is not None:
        if len(points) > max_points:
            idx    = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            if point_colors is not None:
                point_colors = point_colors[idx]
        clr = point_colors
        if clr is not None:
            clr = np.asarray(clr, dtype=float)
            if clr.max() > 1.0:
                clr = clr / 255.0
        traces.append(_plotly_scatter3d(points, clr, size=1.0, opacity=0.4, name="points"))

    layout = _blank_3d_layout()
    layout.title = dict(text=title, font=dict(color="white"))
    fig = go.Figure(data=traces, layout=layout)

    if save_html:
        os.makedirs(os.path.dirname(os.path.abspath(save_html)), exist_ok=True)
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Trajectory plot (pred vs GT)
# ---------------------------------------------------------------------------

def plot_trajectory(
    pred_extrinsics: np.ndarray,
    gt_extrinsics:   Optional[np.ndarray] = None,
    title:           str = "Camera Trajectory",
    save_path:       Optional[str] = None,
    show:            bool = True,
):
    """2-D top-down (XZ) trajectory plot with matplotlib."""
    import matplotlib.pyplot as plt

    def centres(E):
        R = E[:, :3, :3]
        t = E[:, :3,  3]
        return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)

    pred_c = centres(pred_extrinsics)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pred_c[:, 0], pred_c[:, 2], "b-o", ms=3, lw=1.5, label="Predicted")
    ax.plot(pred_c[0, 0],  pred_c[0, 2],  "bs", ms=8)   # start

    if gt_extrinsics is not None:
        gt_c = centres(gt_extrinsics)
        ax.plot(gt_c[:, 0], gt_c[:, 2], "g-o", ms=3, lw=1.5, label="Ground Truth")
        ax.plot(gt_c[0, 0],  gt_c[0, 2],  "gs", ms=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Benchmark chart: memory / time vs number of frames
# ---------------------------------------------------------------------------

def plot_memory_vs_frames(
    benchmark_results: List[dict],
    title:    str = "GPU Memory vs. Frame Count",
    save_path: Optional[str] = None,
    show:      bool = True,
):
    """
    Plot GPU memory and inference time against number of frames.

    benchmark_results: list of dicts from metrics.benchmark_inference
    """
    import matplotlib.pyplot as plt

    ns      = [r["n_frames"]    for r in benchmark_results]
    mems    = [r["peak_gpu_mb"] / 1024 for r in benchmark_results]   # GB
    times   = [r["time_mean_s"] for r in benchmark_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ns, mems, "b-o", lw=2, ms=6)
    ax1.set_xlabel("Number of frames")
    ax1.set_ylabel("Peak GPU memory (GB)")
    ax1.set_title("GPU Memory Usage")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=16, color="r", ls="--", label="16 GB limit (T4/P100)")
    ax1.legend()

    ax2.plot(ns, times, "g-o", lw=2, ms=6)
    ax2.set_xlabel("Number of frames")
    ax2.set_ylabel("Inference time (s)")
    ax2.set_title("Inference Time")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Chunk-coloured point cloud (Phase 3)
# ---------------------------------------------------------------------------

def plot_chunk_alignment(
    chunks_points:   List[np.ndarray],
    chunks_extrinsics: List[np.ndarray],
    title: str = "Chunk-wise Reconstruction",
    max_points_per_chunk: int = 20_000,
    save_html: Optional[str] = None,
    show: bool = True,
):
    """
    Visualise multiple chunks each in a distinct colour, after alignment.

    Args:
        chunks_points:     list of (N_i, 3) point arrays in common world frame
        chunks_extrinsics: list of (K_i, 3, 4) extrinsics in common world frame
    """
    import plotly.graph_objects as go

    palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62",
    ]

    traces = []
    for idx, (pts, extrs) in enumerate(zip(chunks_points, chunks_extrinsics)):
        color = palette[idx % len(palette)]

        if len(pts) > max_points_per_chunk:
            sub = np.random.choice(len(pts), max_points_per_chunk, replace=False)
            pts = pts[sub]

        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name=f"chunk {idx}",
            marker=dict(size=1.2, color=color, opacity=0.6),
        ))

        if extrs is not None and len(extrs) > 0:
            R = extrs[:, :3, :3]
            t = extrs[:, :3,  3]
            centers = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
            traces.append(go.Scatter3d(
                x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode="markers",
                name=f"cams {idx}",
                marker=dict(size=5, color=color, symbol="diamond"),
                showlegend=False,
            ))

    layout = _blank_3d_layout()
    layout.title = dict(text=title, font=dict(color="white"))
    fig = go.Figure(data=traces, layout=layout)

    if save_html:
        os.makedirs(os.path.dirname(os.path.abspath(save_html)), exist_ok=True)
        fig.write_html(save_html)
    if show:
        fig.show()
    return fig
