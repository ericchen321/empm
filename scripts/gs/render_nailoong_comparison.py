"""Render one frame-0 Nailoong Gaussian from two saved EMPM trajectories.

The observed panel is actual per-frame RGB under the object's observed mask.
Both prediction panels deform the same initial Gaussian and use the same
calibrated camera, light-grey background, and fixed Gaussian-to-track mapping.
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from third_party.gaussian_splatting.gaussian_renderer import render
from third_party.gaussian_splatting.scene.gaussian_model import GaussianModel
from scripts.gs.nailoong_object_mask import load_object_mask
from scripts.gs.nailoong_render_utils import load_case_cameras


BACKGROUND = np.array([0.93, 0.93, 0.93], dtype=np.float32)
PART_NEIGHBORS = 16
GAUSSIAN_ANCHORS = 4


def motion_mapping(gaussian_xyz, initial_tracks, part_labels):
    """Attach each Gaussian to nearby tracks of its nearest semantic part."""
    _, nearest_track = cKDTree(initial_tracks).query(gaussian_xyz)
    gaussian_parts = part_labels[nearest_track]
    gaussian_anchors = np.empty((len(gaussian_xyz), GAUSSIAN_ANCHORS), dtype=np.int32)
    anchor_weights = np.empty((len(gaussian_xyz), GAUSSIAN_ANCHORS), dtype=np.float32)
    track_neighbors = np.empty((len(initial_tracks), PART_NEIGHBORS), dtype=np.int32)
    for part in np.unique(part_labels):
        track_ids = np.flatnonzero(part_labels == part)
        tree = cKDTree(initial_tracks[track_ids])
        _, neighbor_local = tree.query(initial_tracks[track_ids], k=PART_NEIGHBORS + 1)
        track_neighbors[track_ids] = track_ids[neighbor_local[:, 1:]]

        gaussian_ids = np.flatnonzero(gaussian_parts == part)
        distances, local = tree.query(gaussian_xyz[gaussian_ids], k=GAUSSIAN_ANCHORS)
        gaussian_anchors[gaussian_ids] = track_ids[local]
        weights = 1.0 / (distances + 1e-6)
        anchor_weights[gaussian_ids] = weights / weights.sum(axis=1, keepdims=True)
    return gaussian_anchors, anchor_weights, track_neighbors


def local_rotations(initial_tracks, current_tracks, neighbors):
    """Fit a rigid rotation around every tracked point within its part."""
    initial_adj = initial_tracks[neighbors]
    current_adj = current_tracks[neighbors]
    a = initial_adj - initial_adj.mean(axis=1, keepdims=True)
    b = current_adj - current_adj.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", a, b)
    u, _, vt = np.linalg.svd(covariance)
    handedness = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1))
    vt[handedness < 0, -1, :] *= -1
    return vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)


def quaternion_multiply(a, b):
    result = np.empty_like(a)
    result[:, 0] = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
    result[:, 1:] = (
        a[:, :1] * b[:, 1:]
        + b[:, :1] * a[:, 1:]
        + np.cross(a[:, 1:], b[:, 1:])
    )
    return result


def deform_gaussians(initial_xyz, initial_quaternion, initial_tracks,
                     current_tracks, anchors, weights, neighbors):
    if current_tracks is initial_tracks:
        return initial_xyz, initial_quaternion
    rotations = local_rotations(initial_tracks, current_tracks, neighbors)
    offsets = initial_xyz[:, None, :] - initial_tracks[anchors]
    moved = (
        np.einsum("nkij,nkj->nki", rotations[anchors], offsets)
        + current_tracks[anchors]
    )
    xyz = np.sum(moved * weights[:, :, None], axis=1).astype(np.float32)

    xyzw = Rotation.from_matrix(rotations).as_quat().astype(np.float32)
    local_quaternion = xyzw[:, [3, 0, 1, 2]]
    local_quaternion[local_quaternion[:, 0] < 0] *= -1
    blended = np.sum(local_quaternion[anchors] * weights[:, :, None], axis=1)
    blended /= np.linalg.norm(blended, axis=1, keepdims=True)
    quaternion = quaternion_multiply(blended, initial_quaternion).astype(np.float32)
    return xyz, quaternion


def render_object(model, camera, background, xyz, quaternion, observed_mask,
                  observed_depth_m):
    model._xyz.data.copy_(torch.from_numpy(xyz).to("cuda"))
    model._rotation.data.copy_(torch.from_numpy(quaternion).to("cuda"))
    result = render(camera, model, None, background, use_gsplat=True)
    rgba = result["render"].permute(1, 2, 0).detach().cpu().numpy()
    color = rgba[..., :3].copy()
    rendered_depth = result["depth"].detach().cpu().numpy()
    # Observed foreground grippers can cover the object; they use the common
    # background in all three panels because this comparison shows object only.
    closer_nonobject = (
        ~observed_mask
        & (observed_depth_m > 0.2)
        & (observed_depth_m < rendered_depth - 0.001)
        & (rgba[..., 3] > 0.05)
    )
    color[closer_nonobject] = BACKGROUND
    return Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255 + 0.5))


def observed_panel(case_dir, camera_index, frame):
    rgb = np.asarray(Image.open(case_dir / "color" / str(camera_index) / f"{frame}.png").convert("RGB"))
    mask = load_object_mask(case_dir, camera_index, frame)
    depth_m = np.load(case_dir / "depth" / str(camera_index) / f"{frame}.npy") / 1000.0
    object_only = np.full_like(rgb, np.uint8(BACKGROUND * 255 + 0.5))
    object_only[mask] = rgb[mask]
    return Image.fromarray(object_only), mask, depth_m


def comparison_panel(observed, default, semantic, frame, fps, train_end, camera_name,
                     slow=False):
    width, height = observed.size
    canvas = Image.new("RGB", (width * 3, height + 68), tuple(np.uint8(BACKGROUND * 255 + 0.5)))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    phase = "INITIAL" if frame == 0 else ("TRAIN" if frame <= train_end else "HELD OUT")
    header = f"Nailoong | {camera_name} | frame {frame:02d}/69 | t={frame/fps:.2f}s | {phase}"
    if slow:
        header += " | 10x SLOW PLAYBACK (3 fps)"
    draw.text((12, 6), header, fill=(25, 25, 25), font=font)
    for index, (label, panel) in enumerate((
        ("OBSERVED RGB | object mask", observed),
        ("DEFAULT EMPM | 8 evaluations", default),
        ("SEMANTIC EMPM | 8 evaluations", semantic),
    )):
        draw.text((index * width + 12, 38), label, fill=(25, 25, 25), font=small)
        canvas.paste(panel, (index * width, 68))
    return canvas


def single_panel(panel, label, frame, fps, train_end, camera_name, slow=False):
    width, height = panel.size
    canvas = Image.new("RGB", (width, height + 68), tuple(np.uint8(BACKGROUND * 255 + 0.5)))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    phase = "INITIAL" if frame == 0 else ("TRAIN" if frame <= train_end else "HELD OUT")
    draw.text((12, 6), f"{camera_name} | frame {frame:02d}/69 | t={frame/fps:.2f}s | {phase}", fill=(25, 25, 25), font=font)
    caption = label + (" | 10x SLOW PLAYBACK (3 fps)" if slow else "")
    draw.text((12, 38), caption, fill=(25, 25, 25), font=font)
    canvas.paste(panel, (0, 68))
    return canvas


def diagnostic_panel(observed, driven, frame, fps, camera_name):
    width, height = observed.size
    canvas = Image.new("RGB", (width * 2, height + 68), tuple(np.uint8(BACKGROUND * 255 + 0.5)))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    draw.text((12, 6), f"{camera_name} | frame {frame:02d}/69 | t={frame/fps:.2f}s | GT-track diagnostic", fill=(25, 25, 25), font=font)
    draw.text((12, 38), "OBSERVED RGB | object mask", fill=(25, 25, 25), font=font)
    draw.text((width + 12, 38), "Frame-0 Gaussian driven by observed 3D tracks", fill=(25, 25, 25), font=font)
    canvas.paste(observed, (0, 68))
    canvas.paste(driven, (width, 68))
    return canvas


def encode_video(frames_dir, output_path, first_frame, frame_count, fps):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-start_number", str(first_frame),
        "-i", str(frames_dir / "frame_%03d.png"),
        "-frames:v", str(frame_count), "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("outputs/nailoong_gaussian_videos/appearance/point_cloud.ply"))
    parser.add_argument("--case-dir", type=Path, default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/nailoong_material_init/experiment_refined"))
    parser.add_argument("--field-path", type=Path, default=Path("outputs/nailoong_material_init/field.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nailoong_gaussian_videos/comparison"))
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--frames", type=int, nargs="+", help="Render selected stills for inspection; omit for all 70 frames and videos")
    parser.add_argument("--diagnostic-gt", action="store_true", help="Also drive the frame-0 Gaussian with observed 3D tracks for diagnostic stills")
    args = parser.parse_args()

    metadata = json.loads((args.case_dir / "metadata.json").read_text())
    with (args.case_dir / "final_data.pkl").open("rb") as stream:
        observed_tracks = np.asarray(pickle.load(stream)["object_points"], dtype=np.float32)
    initial_tracks = observed_tracks[0]
    predictions = np.load(args.run_dir / "predictions.npz")
    trajectories = {
        "default": predictions["default_eval8_world_m"],
        "semantic": predictions["spatial_eval8_world_m"],
    }
    part_labels = np.load(args.field_path)["part_labels"]
    model = GaussianModel(0)
    model.load_ply(str(args.model_path))
    initial_xyz = model.get_xyz.detach().cpu().numpy().copy()
    initial_quaternion = model.get_rotation.detach().cpu().numpy().copy()
    anchors, weights, neighbors = motion_mapping(initial_xyz, initial_tracks, part_labels)
    cameras = load_case_cameras(args.case_dir, args.cameras)
    background = torch.tensor(BACKGROUND, device="cuda")
    frames = sorted(set(args.frames)) if args.frames else range(metadata["frame_num"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for frame in frames:
            deformed = {}
            for name, trajectory in trajectories.items():
                deformed[name] = deform_gaussians(
                    initial_xyz, initial_quaternion, initial_tracks,
                    initial_tracks if frame == 0 else trajectory[frame],
                    anchors, weights, neighbors,
                )
            for camera_index, camera in cameras.items():
                observed, mask, depth_m = observed_panel(args.case_dir, camera_index, frame)
                rendered = {
                    name: render_object(model, camera, background, xyz, quaternion, mask, depth_m)
                    for name, (xyz, quaternion) in deformed.items()
                }
                panels = {
                    "gt": ("OBSERVED RGB | object mask", observed),
                    "default": ("DEFAULT EMPM | 8 evaluations", rendered["default"]),
                    "semantic": ("SEMANTIC EMPM | 8 evaluations", rendered["semantic"]),
                }
                camera_dir = args.output_dir / f"camera_{camera_index}"
                panel = comparison_panel(
                    observed, rendered["default"], rendered["semantic"],
                    frame, metadata["fps"], 63, metadata["camera_names"][camera_index],
                )
                frames_dir = camera_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                panel.save(frames_dir / f"frame_{frame:03d}.png")
                for name, (label, image) in panels.items():
                    single_dir = camera_dir / name / "frames"
                    single_dir.mkdir(parents=True, exist_ok=True)
                    single_panel(image, label, frame, metadata["fps"], 63,
                                 metadata["camera_names"][camera_index]).save(
                        single_dir / f"frame_{frame:03d}.png"
                    )
                if args.diagnostic_gt:
                    gt_xyz, gt_quaternion = deform_gaussians(
                        initial_xyz, initial_quaternion, initial_tracks,
                        initial_tracks if frame == 0 else observed_tracks[frame],
                        anchors, weights, neighbors,
                    )
                    gt_driven = render_object(
                        model, camera, background, gt_xyz, gt_quaternion, mask, depth_m,
                    )
                    diagnostic_panel(
                        observed, gt_driven, frame, metadata["fps"],
                        metadata["camera_names"][camera_index],
                    ).save(camera_dir / f"diagnostic_gt_tracks_frame_{frame:03d}.png")
                if frame >= 64:
                    slow_dir = camera_dir / "heldout_frames"
                    slow_dir.mkdir(parents=True, exist_ok=True)
                    comparison_panel(
                        observed, rendered["default"], rendered["semantic"],
                        frame, metadata["fps"], 63, metadata["camera_names"][camera_index],
                        slow=True,
                    ).save(slow_dir / f"frame_{frame:03d}.png")
                    for name, (label, image) in panels.items():
                        slow_single_dir = camera_dir / name / "heldout_frames"
                        slow_single_dir.mkdir(parents=True, exist_ok=True)
                        single_panel(image, label, frame, metadata["fps"], 63,
                                     metadata["camera_names"][camera_index], slow=True).save(
                            slow_single_dir / f"frame_{frame:03d}.png"
                        )
            print(f"Rendered frame {frame:02d}/{metadata['frame_num'] - 1}", flush=True)

    if args.frames:
        return
    for camera_index in cameras:
        camera_dir = args.output_dir / f"camera_{camera_index}"
        for name in ("comparison", "gt", "default", "semantic"):
            source_dir = camera_dir if name == "comparison" else camera_dir / name
            encode_video(source_dir / "frames", args.output_dir / f"camera_{camera_index}_{name}_full_30fps.mp4", 0, metadata["frame_num"], metadata["fps"])
            encode_video(source_dir / "heldout_frames", args.output_dir / f"camera_{camera_index}_{name}_heldout_3fps.mp4", 64, 6, 3)


if __name__ == "__main__":
    main()
