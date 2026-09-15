"""
Module: Depth Anything 3 - VSLAM-LAB entry point (mono, DA3-Streaming)
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2026-09-15
- Updated: 2026-09-15
- License: Apache-2.0 (Depth Anything 3 / DA3-Streaming, adapted from VGGT-Long)

Runs upstream's da3_streaming/da3_streaming.py on a VSLAM-LAB sequence: every frame of the
experiment's rgb csv (in csv order) goes through DA3 in overlapping chunks, consecutive chunks are
Sim(3)-aligned on their overlap, SALAD retrieval proposes loop closures that a Sim(3) pose graph
closes, and the per-frame camera-to-world poses land in <exp_folder>/<exp_it>_KeyFrameTrajectory.csv.
Upstream's class takes an image folder and globs it; the subclass below feeds it the csv frame list
so timestamps map one to one. Intermediate results live in <exp_folder>/<exp_it>_da3_streaming/.
"""

import argparse
import functools
import gc
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

DA3_STREAMING_DIR = Path(__file__).resolve().parent / "da3_streaming"
sys.path.insert(0, str(DA3_STREAMING_DIR))  # da3_streaming.py imports `loop_utils` relative to its own folder

from da3_streaming import DA3_Streaming  # noqa: E402  (the module file, found before the folder of the same name)
from loop_utils.config_utils import load_config  # noqa: E402
from loop_utils.sim3utils import merge_ply_files, warmup_numba  # noqa: E402

from vslamlab_da3_common import (ensure_salad_checkpoint, load_frames, load_settings, resolve_model_id,  # noqa: E402
                                 show_in_viser, write_trajectory_csv, MODELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DA3-Streaming (VSLAM-LAB entry point, mono)")

    # VSLAM-LAB fixed arguments (BaselineVSLAMLAB.build_execute_command)
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True, help="unused: DA3 predicts its own intrinsics")
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None, help="cam_mono (rgb stream) and a `streaming:` block merged over base_config.yaml")
    parser.add_argument("--verbose", type=str, default="0", help="1 opens the viser viewer with the fused point cloud and the trajectory")
    parser.add_argument("--mode", type=str, default="mono", choices=["mono"], help="VSLAM-LAB mode (selects this entry point; only mono is supported)")

    # DA3-Streaming (da3_streaming/configs/base_config.yaml)
    parser.add_argument("--model", type=str, default="nested-giant-large-1.1", help=f"one of {sorted(MODELS)} or a Hugging Face id")
    parser.add_argument("--chunk_size", type=int, default=120, help="frames per chunk (upstream default 120; memory scales with it)")
    parser.add_argument("--overlap", type=int, default=60, help="frames shared by consecutive chunks (upstream default: half the chunk)")
    parser.add_argument("--loop_enable", type=int, default=1, help="1: SALAD loop detection + Sim(3) pose-graph optimisation")
    parser.add_argument("--process_res", type=int, default=504, help="processing resolution passed to DepthAnything3.inference")
    return parser


class VSLAMLabStreaming(DA3_Streaming):
    """DA3_Streaming fed with an explicit frame list instead of a globbed folder."""

    def __init__(self, image_paths: list[str], save_dir: str, config: dict, process_res: int):
        super().__init__(image_dir=str(Path(image_paths[0]).parent), save_dir=save_dir, config=config)
        self.image_paths = image_paths
        # process_single_chunk calls self.model.inference(images, ref_view_strategy=...) with upstream's default resolution
        self.model.inference = functools.partial(self.model.inference, process_res=process_res)

    def run(self):
        self.img_list = list(self.image_paths)
        print(f"Found {len(self.img_list)} images (from the experiment's rgb csv)")
        if self.loop_enable:
            self.loop_detector.image_paths = [Path(p) for p in self.img_list]  # skips LoopDetector.get_image_paths' glob
        self.process_long_sequence()


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_config(args: argparse.Namespace, settings: dict) -> dict:
    from huggingface_hub import hf_hub_download
    config = load_config(str(DA3_STREAMING_DIR / "configs" / "base_config.yaml"))
    deep_update(config, settings.get("streaming", {}))
    model_id = resolve_model_id(args.model)
    if Path(model_id).is_dir():
        weights, model_config = Path(model_id) / "model.safetensors", Path(model_id) / "config.json"
    else:
        weights = hf_hub_download(model_id, "model.safetensors")
        model_config = hf_hub_download(model_id, "config.json")
    config["Weights"] = {"DA3": str(weights), "DA3_CONFIG": str(model_config), "SALAD": ensure_salad_checkpoint()}
    config["Model"]["chunk_size"] = args.chunk_size
    config["Model"]["overlap"] = args.overlap
    config["Model"]["loop_enable"] = bool(args.loop_enable)
    return config


def main() -> None:
    args = build_parser().parse_args()
    vis = bool(int(args.verbose))
    settings = load_settings(args.settings_yaml)
    if args.overlap >= args.chunk_size:
        raise SystemExit(f"overlap ({args.overlap}) must be smaller than chunk_size ({args.chunk_size})")

    image_names, timestamps = load_frames(args.sequence_path, args.rgb_csv, settings.get("cam_mono", "rgb_0"))
    config = build_config(args, settings)
    run_dir = args.exp_folder / f"{args.exp_it.zfill(5)}_da3_streaming"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    print(f"DA3-Streaming: {len(image_names)} frames, chunk_size={args.chunk_size}, overlap={args.overlap}, "
          f"loop_enable={bool(args.loop_enable)}, model={resolve_model_id(args.model)}, run dir {run_dir}")

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    t0 = time.time()
    streaming = VSLAMLabStreaming(image_names, str(run_dir), config, process_res=args.process_res)
    streaming.run()
    streaming.close()
    print(f"DA3-Streaming took {time.time() - t0:.1f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del streaming
    torch.cuda.empty_cache()
    gc.collect()

    # save_camera_poses wrote one flattened 4x4 camera-to-world matrix per frame, in img_list (= csv) order
    cam_to_world = np.loadtxt(run_dir / "camera_poses.txt").reshape(-1, 4, 4)
    assert len(cam_to_world) == len(timestamps), f"{len(cam_to_world)} poses for {len(timestamps)} frames"
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    write_trajectory_csv(keyframe_csv, timestamps, cam_to_world)
    print(f"Trajectory written to {keyframe_csv}")

    if vis:
        import trimesh
        combined = run_dir / "pcd" / "combined_pcd.ply"
        merge_ply_files(str(run_dir / "pcd"), str(combined))
        cloud = trimesh.load(str(combined), process=False)
        points = np.asarray(cloud.vertices, dtype=np.float32)
        colors = np.asarray(cloud.colors)[:, :3].astype(np.uint8) if hasattr(cloud, "colors") and len(cloud.colors) else np.full((len(points), 3), 200, np.uint8)
        show_in_viser(points, colors, cam_to_world)


if __name__ == "__main__":
    main()
