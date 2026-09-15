"""
Module: Depth Anything 3 - VSLAM-LAB entry point (mono, single window)
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: Apache-2.0 (Depth Anything 3)

One feed-forward DA3 pass over every frame of the experiment's rgb csv (the framework's rgb_max /
rgb_step / rgb_placecell keys decide which frames those are), camera-to-world poses written to
<exp_folder>/<exp_it>_KeyFrameTrajectory.csv, optional viser visualization. With the nested model the
poses and depth come out in metres. Same shape as the vggt / vggt-omega entry points.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from vslamlab_da3_common import (load_frames, load_settings, resolve_model_id, show_in_viser, unproject_depth,
                                 w2c_to_c2w, write_trajectory_csv, MODELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth Anything 3 (VSLAM-LAB entry point, mono, single window)")

    # VSLAM-LAB fixed arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True, help="unused: DA3 predicts its own intrinsics")
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None, help="cam_mono (rgb stream)")
    parser.add_argument("--verbose", type=str, default="0", help="1 opens the viser viewer with the reconstruction")
    parser.add_argument("--mode", type=str, default="mono", choices=["mono"], help="VSLAM-LAB mode (selects this entry point; only mono is supported)")
    parser.add_argument("--rgb_max", type=int, default=None, help="consumed by VSLAM-LAB (Run/run_functions.py) when writing rgb_csv; accepted here so the command line parses")

    # Depth Anything 3 (api.DepthAnything3.inference)
    parser.add_argument("--model", type=str, default="nested-giant-large-1.1", help=f"one of {sorted(MODELS)} or a Hugging Face id")
    parser.add_argument("--use_ray_pose", type=int, default=0, help="1: derive poses from the ray head (slower, upstream reports higher AUC); 0: camera decoder")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced", choices=["first", "middle", "saddle_balanced", "saddle_sim_range"])
    parser.add_argument("--process_res", type=int, default=504, help="processing resolution (upstream default 504)")
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize", choices=["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"])
    parser.add_argument("--conf_threshold", type=float, default=40.0, help="viewer only: percentage of lowest-confidence points hidden")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    vis = bool(int(args.verbose))
    settings = load_settings(args.settings_yaml)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = resolve_model_id(args.model)
    print(f"Loading Depth Anything 3 model {model_id}...")
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()

    image_names, timestamps = load_frames(args.sequence_path, args.rgb_csv, settings.get("cam_mono", "rgb_0"))
    print(f"Running inference on {len(image_names)} images from {args.sequence_path} "
          f"(process_res={args.process_res}, use_ray_pose={bool(args.use_ray_pose)}, ref_view_strategy={args.ref_view_strategy})...")
    t0 = time.time()
    with torch.inference_mode():
        prediction = model.inference(
            image_names,
            use_ray_pose=bool(args.use_ray_pose),
            ref_view_strategy=args.ref_view_strategy,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"Inference took {time.time() - t0:.2f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
              f"processed images {prediction.processed_images.shape}, metric={bool(prediction.is_metric)}")

    # DA3 extrinsics are camera-from-world (OpenCV convention); the trajectory wants camera-to-world
    cam_to_world = w2c_to_c2w(np.asarray(prediction.extrinsics, dtype=np.float64))
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    write_trajectory_csv(keyframe_csv, timestamps, cam_to_world)
    print(f"Trajectory written to {keyframe_csv}")

    if vis:
        depth = np.asarray(prediction.depth, dtype=np.float32)
        points = unproject_depth(depth, np.asarray(prediction.extrinsics), np.asarray(prediction.intrinsics)).reshape(-1, 3)
        colors = np.asarray(prediction.processed_images).reshape(-1, 3)
        conf = np.asarray(prediction.conf).reshape(-1) if prediction.conf is not None else np.ones(len(points))
        keep = (conf >= np.percentile(conf, args.conf_threshold)) & np.isfinite(points).all(axis=1) & (depth.reshape(-1) > 0)
        show_in_viser(points[keep], colors[keep], cam_to_world, images=prediction.processed_images, intrinsics=prediction.intrinsics)


if __name__ == "__main__":
    main()
