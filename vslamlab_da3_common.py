"""
Module: Depth Anything 3 - VSLAM-LAB shared helpers
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: Apache-2.0 (Depth Anything 3)

Shared by vslamlab_da3.py (single window) and vslamlab_da3_streaming.py (chunked full sequence):
model name table, settings / frame loading, trajectory csv writer, SALAD checkpoint, viser viewer.
"""

from __future__ import annotations

import os
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

# `model` parameter -> Hugging Face repo (weights are not gated). The -1.1 checkpoints are the
# retrained ones upstream recommends; the older names stay selectable for comparisons.
MODELS = {
    "nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",  # any-view giant + metric model: metric depth and poses
    "giant-1.1": "depth-anything/DA3-GIANT-1.1",
    "large-1.1": "depth-anything/DA3-LARGE-1.1",
    "base": "depth-anything/DA3-BASE",
    "small": "depth-anything/DA3-SMALL",
    "nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "giant": "depth-anything/DA3-GIANT",
    "large": "depth-anything/DA3-LARGE",
}

SALAD_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


def resolve_model_id(model: str) -> str:
    """Short name from MODELS, or a full Hugging Face id / local directory passed through."""
    if model in MODELS:
        return MODELS[model]
    if "/" in model:
        return model
    raise SystemExit(f"unknown model '{model}'; expected one of {sorted(MODELS)} or a Hugging Face id")


def load_settings(settings_yaml: Path | None) -> dict:
    if settings_yaml is None or not Path(settings_yaml).is_file():
        return {}
    with open(settings_yaml, "r") as f:
        return yaml.safe_load(f) or {}


def load_frames(sequence_path: Path, rgb_csv: Path, cam_name: str) -> tuple[list[str], list[int]]:
    """Frame paths and timestamps (ns) of the mono camera, in csv order."""
    df = pd.read_csv(rgb_csv)
    image_names = [str(Path(sequence_path) / p) for p in df[f"path_{cam_name}"]]
    timestamps = df[f"ts_{cam_name} (ns)"].astype("int64").tolist()
    return image_names, timestamps


def write_trajectory_csv(keyframe_csv: Path, timestamps: list[int], cam_to_world: np.ndarray) -> None:
    """VSLAM-LAB trajectory: camera-to-world poses (N, 4, 4) with their timestamps."""
    quaternions = R.from_matrix(cam_to_world[:, :3, :3]).as_quat()  # x, y, z, w
    rows = [[ts, *cam_to_world[i, :3, 3], *quaternions[i]] for i, ts in enumerate(timestamps)]
    pd.DataFrame(rows, columns=["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]).to_csv(keyframe_csv, index=False)


def w2c_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    """DA3 extrinsics are camera-from-world (N, 3, 4) or (N, 4, 4), OpenCV convention -> camera-to-world (N, 4, 4)."""
    w2c = np.tile(np.eye(4), (len(extrinsics), 1, 1))
    w2c[:, :3, :] = extrinsics[:, :3, :]
    return np.linalg.inv(w2c)


def ensure_salad_checkpoint() -> str:
    """dino_salad.ckpt in the torch hub cache (shared with vggtslam), downloaded when missing."""
    import torch
    ckpt_path = os.path.join(torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
    if not (os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.hub.download_url_to_file(SALAD_CKPT_URL, ckpt_path)
    return ckpt_path


def unproject_depth(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Depth (N,H,W) + camera-from-world (N,3,4|4,4) + K (N,3,3) -> world points (N,H,W,3)."""
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx, fy = intrinsics[:, 0, 0][:, None, None], intrinsics[:, 1, 1][:, None, None]
    cx, cy = intrinsics[:, 0, 2][:, None, None], intrinsics[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation, translation = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), camera_points - translation[:, None, None, :])


def show_in_viser(points: np.ndarray, colors: np.ndarray, cam_to_world: np.ndarray, port: int = 8080,
                  images: np.ndarray | None = None, intrinsics: np.ndarray | None = None,
                  first_client_timeout_s: float = 60.0) -> None:
    """Point cloud + cameras in viser; opens the browser and returns once every client has disconnected
    (or when nobody connected within first_client_timeout_s, for headless machines)."""
    import viser
    import viser.transforms as viser_tf

    center = points.mean(axis=0) if len(points) else np.zeros(3)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_point_cloud("points", points=(points - center).astype(np.float32), colors=colors, point_size=0.002, point_shape="circle")
    for i, pose in enumerate(cam_to_world):
        T = viser_tf.SE3.from_matrix(pose)
        if images is not None and intrinsics is not None:
            img = images[i]
            h, w = img.shape[:2]
            fov = 2 * np.arctan2(h / 2, intrinsics[i, 1, 1])
            server.scene.add_camera_frustum(f"frame_{i}", fov=fov, aspect=w / h, scale=0.05, image=img,
                                            wxyz=T.rotation().wxyz, position=T.translation() - center)
        else:
            server.scene.add_frame(f"frame_{i}", wxyz=T.rotation().wxyz, position=T.translation() - center,
                                   axes_length=0.05, axes_radius=0.002, origin_radius=0.003)

    url = f"http://localhost:{server.get_port()}"
    print(f"Viser viewer at {url}")
    webbrowser.open(url)
    t0 = time.time()
    had_clients = False
    try:
        while True:
            num_clients = len(server.get_clients())
            had_clients = had_clients or num_clients > 0
            if had_clients and num_clients == 0:
                print("(viser) All clients disconnected. Shutting down server.")
                return
            if not had_clients and time.time() - t0 > first_client_timeout_s:
                print(f"(viser) No client connected within {first_client_timeout_s:.0f} s. Shutting down server.")
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        return
