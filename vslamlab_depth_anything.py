"""
VSLAM-LAB capability: Depth Anything 3 monocular depth (entry point in github.com/VSLAM-LAB/depthanything3).

Predicts a per-frame metric depth map for the rgb_0 frames of the given VSLAM-LAB sequence folders with Depth
Anything 3 (default: the nested DA3NESTED-GIANT-LARGE-1.1 model, metric). Frames go through the model in batches of
--batch_size (default 8), which DA3 processes jointly as multiple views; each depth map is resized (nearest) back to
its source frame's resolution. Depth is stored as 16-bit PNG with depth (m) = pixel_value / depth_factor (default 256:
~3.9 mm resolution, 256 m max, the same convention as the fastfoundationstereo capability); 0 means invalid.

Artifact (inside each sequence folder): depth_anything_0/<rgb_0 frame stem>.png, plus the completion marker
depth_anything_0/.depth_anything_complete holding `depth_factor: <value>`. Frames whose depth PNG already exists are
skipped (an interrupted run resumes; the remaining frames are batched among themselves); a sequence with the marker
is skipped unless --overwrite. The sequence's rgb.csv and calibration.yaml are never modified. If an rgb_raw.csv
backup exists (VSLAM-LAB's pre-sampling frame list), depth is generated from it, so a downsampled rgb.csv still gets
complete coverage.

VSLAM-LAB (Capabilities/depth_anything.py) resolves dataset/sequence targets into --sequence-path folders, calls this
script through the `depth-inference` pixi task and wires the depth into experiments ('depth: depth_anything').
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"  # the retrained -1.1 checkpoints supersede the originals
DEPTH_FOLDER_BASE = "depth_anything"
COMPLETE_MARKER = ".depth_anything_complete"
DEPTH_FACTOR = 256.0

SCRIPT_LABEL = "\033[95m[vslamlab_depth_anything.py]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}", flush=True)


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m[warning]\033[0m {msg}", flush=True)


def raw_path(csv_path: Path) -> Path:
    """<name>_raw<suffix>: VSLAM-LAB's pre-edit backup of a sequence csv (full frame list before sampling)."""
    return csv_path.with_name(f"{csv_path.stem}_raw{csv_path.suffix}")


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """(header, rows) of a VSLAM-LAB sequence csv, rows sorted by the timestamp column."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    rows.sort(key=lambda row: int(row[0]) if row[0].lstrip("-").isdigit() else float(row[0]))
    return header, rows


def load_model(model_id: str, device: torch.device):
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(model_id).to(device)


def depth_png(depth: np.ndarray, height: int, width: int, depth_factor: float) -> np.ndarray:
    """Model depth (m) -> 16-bit PNG values at the source frame's resolution (0 = invalid)."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    return np.clip(depth * depth_factor, 0, 65535).astype(np.uint16)


def depth_sequence(
    seq_path: Path, model_loader, *, batch_size: int, depth_factor: float = DEPTH_FACTOR,
    depth_folder_base: str = DEPTH_FOLDER_BASE, overwrite: bool = False,
) -> None:
    rgb_csv = seq_path / "rgb.csv"
    rgb_raw = raw_path(rgb_csv)
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv  # full pre-sampling frame list when available
    if not source_csv.exists():
        print_warning(f"Skipping {seq_path} - missing rgb.csv")
        return

    header, rows = read_csv_rows(source_csv)
    if "path_rgb_0" not in header:
        print_warning(f"Skipping {seq_path} - rgb.csv has no path_rgb_0 column")
        return

    depth_dir = seq_path / f"{depth_folder_base}_0"
    marker = depth_dir / COMPLETE_MARKER
    if marker.exists() and not overwrite:
        print_info(f"Skipping {seq_path} - {depth_dir.name} already complete (use --overwrite to recompute)")
        return
    if overwrite and depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    path_idx = header.index("path_rgb_0")
    frames = [seq_path / row[path_idx] for row in rows]
    todo = [f for f in frames if not (depth_dir / f"{f.stem}.png").exists()]  # resume support
    if todo:
        model = model_loader()
        with torch.inference_mode():
            for s in tqdm(range(0, len(todo), batch_size), total=math.ceil(len(todo) / batch_size),
                          desc=f"{seq_path.parent.name}/{seq_path.name} depth"):
                batch = todo[s:s + batch_size]
                pred = model.inference([p.as_posix() for p in batch])
                for frame, depth in zip(batch, pred.depth):
                    height, width = imageio.imread(frame).shape[:2]
                    imageio.imwrite(depth_dir / f"{frame.stem}.png", depth_png(depth, height, width, depth_factor))
                del pred
                torch.cuda.empty_cache()

    # The marker doubles as metadata: VSLAM-LAB reads depth_factor from it when registering the depth stream.
    marker.write_text(f"depth_factor: {float(depth_factor)}\n")
    print_info(f"{seq_path} - wrote {len(todo)} depth maps to {depth_dir}"
               + (f" ({len(frames) - len(todo)} already existed)" if len(todo) < len(frames) else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-frame metric depth (16-bit PNG) for rgb_0 with Depth Anything 3.")
    parser.add_argument("--sequence-path", nargs="+", type=Path, default=[], dest="sequence_paths",
                        help="VSLAM-LAB sequence folders (each with rgb.csv)")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Depth Anything 3 model id (Hugging Face)")
    parser.add_argument("--batch_size", type=int, default=8, help="Frames per DA3 forward pass (processed jointly)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-factor", type=float, default=DEPTH_FACTOR, dest="depth_factor",
                        help="Depth (m) = png_value / depth_factor (default: 256)")
    parser.add_argument("--depth-folder-base", default=DEPTH_FOLDER_BASE, dest="depth_folder_base",
                        help="Depth folder prefix; depth is written to <base>_0 (default: depth_anything)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute depth even if it already exists")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download/cache the model and exit, no sequences needed")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    if args.prefetch:
        load_model(args.model_id, torch.device("cpu"))
        print_info(f"Model ready: {args.model_id}")
        return
    if not args.sequence_paths:
        parser.error("--sequence-path is required (unless --prefetch)")

    model = {}

    def model_loader():  # lazy: complete sequences do not pay the model load
        if not model:
            model["m"] = load_model(args.model_id, device)
        return model["m"]

    for seq_path in args.sequence_paths:
        depth_sequence(seq_path.resolve(), model_loader, batch_size=args.batch_size, depth_factor=args.depth_factor,
                       depth_folder_base=args.depth_folder_base, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
