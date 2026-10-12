"""Fit one shared Gaussian appearance from the three initial RGB-D views."""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from third_party.gaussian_splatting.gaussian_renderer import render
from third_party.gaussian_splatting.scene.gaussian_model import GaussianModel
from third_party.gaussian_splatting.utils.graphics_utils import BasicPointCloud
from third_party.gaussian_splatting.utils.sh_utils import RGB2SH
from scripts.gs.nailoong_render_utils import load_case_cameras
from scripts.gs.nailoong_object_mask import load_object_mask


def initial_observations(case_dir):
    metadata = json.loads((case_dir / "metadata.json").read_text())
    with (case_dir / "calibrate.pkl").open("rb") as f:
        c2ws = pickle.load(f)
    with (case_dir / "final_data.pkl").open("rb") as f:
        tracks = pickle.load(f)["object_points"][0]
    tree = cKDTree(tracks)
    width, height = metadata["WH"]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    observations, all_points, all_colors = {}, [], []
    for camera in range(len(c2ws)):
        rgb = cv2.cvtColor(cv2.imread(str(case_dir / "color" / str(camera) / "0.png")),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        depth = np.load(case_dir / "depth" / str(camera) / "0.npy") / 1000
        masks = [cv2.imread(str(case_dir / "mask" / str(camera) / str(label) / "0.png"),
                            cv2.IMREAD_GRAYSCALE) > 0 for label in range(3)]
        K = np.asarray(metadata["intrinsics"][camera])
        xyz_camera = np.stack(((u + K[0, 2] - width) * depth / K[0, 0],
                               (v - K[1, 2]) * depth / K[1, 1], depth), axis=-1)
        xyz_world = xyz_camera @ c2ws[camera, :3, :3].T + c2ws[camera, :3, 3]
        mask = load_object_mask(case_dir, camera, 0) & (depth > 0)
        near_tracks = tree.query(xyz_world[mask])[0] < .01
        mask[mask] = near_tracks
        all_points.append(xyz_world[mask])
        all_colors.append(rgb[mask])
        occluder = cv2.dilate((masks[1] | masks[2]).astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
        observations[camera] = {"rgb": rgb, "depth": depth, "mask": mask,
                                "valid": ~occluder}
    xyz = np.concatenate(all_points).astype(np.float32)
    colors = np.concatenate(all_colors).astype(np.float32)
    _, chosen = np.unique(np.round(xyz / .00045).astype(np.int32), axis=0,
                           return_index=True)
    return xyz[chosen], colors[chosen], observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/nailoong_gaussian_videos/appearance"))
    parser.add_argument("--iterations", type=int, default=1800)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    points, colors, observations = initial_observations(args.case_dir)
    cameras = load_case_cameras(args.case_dir, camera_indices=(0, 1, 2))
    gaussians = GaussianModel(sh_degree=0)
    gaussians.create_from_pcd(BasicPointCloud(points, colors, np.zeros_like(points)),
                              [SimpleNamespace(image_name=str(c)) for c in cameras], 1.)
    # RGB-D fixes surface geometry; optimize appearance and splat support only.
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    color_logits = torch.nn.Parameter(torch.logit(
        torch.as_tensor(colors, device="cuda").clamp(1e-4, 1 - 1e-4)))
    with torch.no_grad():
        gaussians._opacity.fill_(0.)
        gaussians._scaling.add_(np.log(.8))
    optimizer = torch.optim.Adam([
        {"params": [color_logits], "lr": .025},
        {"params": [gaussians._opacity], "lr": .025},
        {"params": [gaussians._scaling], "lr": .003},
        {"params": [gaussians._rotation], "lr": .001},
    ])
    background = torch.full((3,), .93, device="cuda")
    for item in observations.values():
        for key, value in list(item.items()):
            item[key] = torch.as_tensor(value, device="cuda")
        item["target"] = torch.where(item["mask"][..., None], item["rgb"], background)
    history = []
    started = time.perf_counter()
    for iteration in range(args.iterations):
        camera = iteration % len(cameras)
        observation = observations[camera]
        result = render(cameras[camera], gaussians, None, background,
                        override_color=color_logits.sigmoid())
        image = result["render"][:3].permute(1, 2, 0)
        alpha = result["render"][3]
        valid, mask = observation["valid"], observation["mask"]
        weights = valid.float() * (1 + 4 * mask.float())
        rgb_loss = ((image - observation["target"]).abs().mean(-1) * weights).sum() / weights.sum()
        alpha_loss = ((alpha - mask.float()).abs() * valid).sum() / valid.sum()
        depth_loss = (result["depth"][mask] - observation["depth"][mask]).abs().mean()
        # Unobserved contact regions otherwise permit centimeter-long splats
        # and bright streaks. Keep the RGB-D surface splats spatially local.
        support_loss = torch.relu(gaussians.get_scaling / .002 - 1).square().mean()
        loss = rgb_loss + .2 * alpha_loss + 5 * depth_loss + .2 * support_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0 or iteration == args.iterations - 1:
            row = {"iteration": iteration + 1, "camera": camera,
                   "rgb_l1": float(rgb_loss.detach()), "alpha_l1": float(alpha_loss.detach()),
                   "depth_l1_m": float(depth_loss.detach()),
                   "support_loss": float(support_loss.detach()), "loss": float(loss.detach())}
            history.append(row)
            print(row, flush=True)
    with torch.no_grad():
        gaussians._features_dc.copy_(RGB2SH(color_logits.sigmoid())[:, None, :])
    gaussians.save_ply(str(args.output_dir / "point_cloud.ply"))
    with (args.output_dir / "training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with torch.no_grad():
        for camera, view in cameras.items():
            prediction = render(view, gaussians, None, background)["render"][:3]
            observation = observations[camera]
            prediction = prediction.permute(1, 2, 0)
            prediction = torch.where(observation["valid"][..., None], prediction, background)
            pair = torch.cat((observation["target"], prediction), dim=1)
            cv2.imwrite(str(args.output_dir / f"initial_gt_render_camera{camera}.png"),
                        cv2.cvtColor((pair.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
    metadata = {"case_dir": str(args.case_dir), "training_frames": [0],
                "training_cameras": list(cameras), "sh_degree": 0,
                "gaussian_count": len(points), "iterations": args.iterations,
                "coordinate_system": "original world coordinates, meters",
                "geometry": "fixed frame-0 RGB-D means; trained color, opacity, scale, rotation",
                "appearance_regularization": "RGB constrained to [0,1]; quadratic scale-support penalty above 2 mm",
                "background_rgb": [.93, .93, .93],
                "elapsed_seconds": time.perf_counter() - started}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
