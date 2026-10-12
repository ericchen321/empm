"""Render fixed-parameter full-sequence rollouts with articulated gripper Gaussians.

GT is driven by saved source FEM states. All three panels share the same
learned object appearance and known rigid-gripper geometry and pose inputs.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.gs.nailoong_render_utils import load_case_cameras
from scripts.gs.render_nailoong_comparison import (
    BACKGROUND, deform_gaussians, encode_video, motion_mapping,
)
from scripts.gs.nailoong_rigid_gaussians import build_rigid_assets, pose_rigid_assets
from third_party.gaussian_splatting.gaussian_renderer import render
from third_party.gaussian_splatting.scene.gaussian_model import GaussianModel
from third_party.gaussian_splatting.utils.sh_utils import RGB2SH


PHASES = ((30, "READY"), (50, "GRASP"), (80, "LIFT 45 mm"),
          (115, "SQUEEZE"), (150, "PULL RING"))
LABELS = {"gt": "GT | source FEM-driven Gaussians",
          "default": "Default initialization | fitted @8",
          "semantic": "Semantic initialization | fitted @8"}


def make_panel(images, names, camera_name, step, time_s, slow=False, start_step=0):
    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images), height + 88), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 16)
    phase = next(name for end, name in PHASES if step < end)
    line = f"{camera_name} | source {step:03d}/149 | t={time_s:.2f}s | {phase}"
    draw.text((12, 5), line, font=font, fill=(25, 25, 25))
    explanation = f"Initialized at source {start_step}; fixed fitted E; continuous rollout; known gripper motion"
    if slow:
        explanation = "3x SLOW PLAYBACK (10 fps) | " + explanation
    if len(images) == 1:
        explanation = ("3x SLOW (10 fps) | " if slow else "") + f"Init at source {start_step} | fixed fitted E"
    draw.text((12, 31), explanation, font=small, fill=(80, 80, 80))
    for index, (panel, name) in enumerate(zip(images, names)):
        draw.text((width * index + 12, 59), LABELS[name], font=font, fill=(25, 25, 25))
        canvas.paste(panel, (width * index, 88))
    return canvas


def merged_model(object_model, rigid):
    model = GaussianModel(0)
    count = len(rigid["local_xyz"])
    to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cuda")
    model._xyz = torch.cat([object_model._xyz.detach(), to_tensor(rigid["local_xyz"])])
    model._rotation = torch.cat([object_model._rotation.detach(), to_tensor(rigid["local_quaternions"])])
    model._features_dc = torch.cat([
        object_model._features_dc.detach(), RGB2SH(to_tensor(rigid["colors"]))[:, None, :],
    ])
    model._features_rest = torch.empty((len(model._xyz), 0, 3), device="cuda")
    model._scaling = torch.cat([object_model._scaling.detach(), to_tensor(np.log(rigid["scales"]))])
    opacity = np.asarray(rigid["opacity"]).reshape(count, 1)
    model._opacity = torch.cat([object_model._opacity.detach(), to_tensor(np.log(opacity / (1 - opacity)))])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6"))
    parser.add_argument("--case-dir", type=Path,
                        default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--source-tracks", type=Path,
                        default=Path("outputs/nailoong_full_sequence/source_tracks.npz"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("outputs/nailoong_full_sequence/rollout/trajectories.npz"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("outputs/nailoong_gaussian_videos/appearance/point_cloud.ply"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/nailoong_full_sequence/videos"))
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--focal-scale", type=float, default=.82,
                        help="Widen cameras 1/2 to keep failed floor-level predictions visible")
    parser.add_argument("--frames", nargs="+", type=int, help="Preview source steps only; no encoding")
    parser.add_argument("--preview-gt", action="store_true", help="Render source alone before simulations finish")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.load(args.source_tracks)
    metadata = json.loads((args.case_dir / "metadata.json").read_text())
    with (args.case_dir / "final_data.pkl").open("rb") as stream:
        initial_tracks = pickle.load(stream)["object_points"][0]
    if args.preview_gt:
        trajectories = {"gt": source["object_points_world_m"]}
        steps, times = source["source_steps"], source["time_s"]
    else:
        predictions = np.load(args.predictions)
        trajectories = {"gt": predictions["gt_world_m"],
                        "default": predictions["default_eval8_world_m"],
                        "semantic": predictions["spatial_eval8_world_m"]}
        steps, times = predictions["source_steps"], predictions["time_s"]
    object_model = GaussianModel(0)
    object_model.load_ply(str(args.model_path))
    initial_xyz = object_model.get_xyz.detach().cpu().numpy().copy()
    initial_rotation = object_model.get_rotation.detach().cpu().numpy().copy()
    mapping = motion_mapping(initial_xyz, initial_tracks, source["part_labels"])
    rigid = build_rigid_assets(args.source)
    model = merged_model(object_model, rigid)
    count = len(initial_xyz)
    cameras = load_case_cameras(args.case_dir, args.cameras)
    for index, camera in cameras.items():
        if index != 0:
            camera.K[0, 0] *= args.focal_scale
            camera.K[1, 1] *= args.focal_scale
    background = torch.tensor(BACKGROUND, device="cuda")
    selected = np.flatnonzero(np.isin(steps, args.frames)) if args.frames else np.arange(len(steps))
    names = list(trajectories)
    source_index = {int(step): index for index, step in enumerate(source["source_steps"])}
    with torch.no_grad():
        for index in selected:
            step = int(steps[index])
            rigid_xyz, rigid_quaternion, rigid_color = pose_rigid_assets(
                rigid, source["rigid_poses"][source_index[step]])
            model._xyz[count:].copy_(torch.from_numpy(rigid_xyz).cuda())
            model._rotation[count:].copy_(torch.from_numpy(rigid_quaternion).cuda())
            model._features_dc[count:].copy_(RGB2SH(torch.from_numpy(rigid_color).cuda())[:, None, :])
            images = {camera: [] for camera in cameras}
            for name, trajectory in trajectories.items():
                xyz, quaternion = deform_gaussians(
                    initial_xyz, initial_rotation, initial_tracks, trajectory[index], *mapping)
                model._xyz[:count].copy_(torch.from_numpy(xyz).cuda())
                model._rotation[:count].copy_(torch.from_numpy(quaternion).cuda())
                for camera_index, camera in cameras.items():
                    pixels = render(camera, model, None, background, use_gsplat=True)["render"]
                    rgb = pixels[:3].permute(1, 2, 0).cpu().numpy()
                    images[camera_index].append(Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + .5)))
            for camera_index, panels in images.items():
                camera_name = metadata["camera_names"][camera_index]
                for slow in (False, True):
                    rate = "slow" if slow else "full"
                    directory = args.output_dir / f"camera_{camera_index}" / f"comparison_{rate}"
                    directory.mkdir(parents=True, exist_ok=True)
                    make_panel(panels, names, camera_name, step, times[index], slow, int(steps[0])).save(
                        directory / f"frame_{index:03d}.png")
                    for panel, name in zip(panels, names):
                        single_dir = args.output_dir / f"camera_{camera_index}" / f"{name}_{rate}"
                        single_dir.mkdir(parents=True, exist_ok=True)
                        make_panel([panel], [name], camera_name, step, times[index], slow, int(steps[0])).save(
                            single_dir / f"frame_{index:03d}.png")
            if index % 15 == 0 or index == selected[-1]:
                print(f"Rendered source step {step} ({index + 1}/{len(steps)})", flush=True)
    if args.frames:
        return
    inventory = []
    for camera in cameras:
        for stream in [*names, "comparison"]:
            for rate, fps in (("full", 30), ("slow", 10)):
                path = args.output_dir / f"camera_{camera}_{stream}_{rate}_{fps}fps.mp4"
                encode_video(args.output_dir / f"camera_{camera}" / f"{stream}_{rate}",
                             path, 0, len(steps), fps)
                inventory.append({"path": path.name, "frame_count": len(steps),
                                  "fps": fps, "duration_s": len(steps) / fps})
    description = {
        "gt": "Source FEM states transported to original track identities and rendered with shared Gaussian appearance; not observed RGB.",
        "object_appearance": str(args.model_path), "object_appearance_reference_source_step": 80,
        "rigid_appearance": "Recorded source mesh surface samples converted to anisotropic Gaussians; articulated with exact recorded poses.",
        "rigid_gaussian_count": len(rigid["local_xyz"]), "object_gaussian_count": count,
        "camera_1_2_focal_scale": args.focal_scale,
        "predictions": str(args.predictions), "videos": inventory,
    }
    (args.output_dir / "video_inventory.json").write_text(json.dumps(description, indent=2) + "\n")
    print(f"Encoded {len(inventory)} videos in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
