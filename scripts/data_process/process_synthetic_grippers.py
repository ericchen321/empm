"""Turn the rendered Nailoong RGB-D sequence into offline EMPM trajectories.

The HSV masks are specific to the amber/cyan grippers and Nailoong appearance
in the synthetic recording. CoTracker supplies point correspondences; observed
depth and calibrated cameras supply the 3D positions.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image


MASK_NAMES = ("nailoong", "orange_gripper", "blue_gripper")
OBJECT_QUERIES_PER_CAMERA = 600
GRIPPER_QUERIES_PER_CAMERA = 40
CONTROLLER_POINTS_PER_GRIPPER = 15


def segment(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    orange = (14 <= h) & (h <= 18) & (s >= 185) & (v >= 35)
    blue = (90 <= h) & (h <= 105) & (s >= 110) & (v >= 35)
    nailoong = (
        ((19 <= h) & (h <= 55) & (s >= 70) & (v >= 30))
        | ((160 <= h) & (h <= 179) & (s >= 55) & (v >= 45))
    ) & ~orange & ~blue
    return np.stack((nailoong, orange, blue))


def choose_queries(mask, count, rng):
    pixels = np.argwhere(mask)
    stride = max(1, int(np.sqrt(len(pixels) / count)))
    grid = pixels[(pixels[:, 0] % stride == 0) & (pixels[:, 1] % stride == 0)]
    if len(grid) < count:
        grid = pixels
    chosen = grid[rng.choice(len(grid), size=min(count, len(grid)), replace=False)]
    return chosen[:, ::-1]  # CoTracker uses (x, y).


def fill_missing(values, visible):
    filled = values.copy()
    frames = np.arange(len(values))
    for point in range(values.shape[1]):
        observed = frames[visible[:, point]]
        for axis in range(values.shape[2]):
            filled[:, point, axis] = np.interp(
                frames, observed, values[observed, point, axis]
            )
    return filled


def track_camera(model, source, output, camera, metadata, c2w, rng):
    frame_count = metadata["frame_num"]
    width, height = metadata["WH"]
    rgb_frames = np.stack([
        np.asarray(Image.open(source / "color" / str(camera) / f"{frame}.png").convert("RGB"))
        for frame in range(frame_count)
    ])

    masks = segment(rgb_frames[0])
    query_pixels = []
    query_labels = []
    for label, mask in enumerate(masks):
        count = OBJECT_QUERIES_PER_CAMERA if label == 0 else GRIPPER_QUERIES_PER_CAMERA
        pixels = choose_queries(mask, count, rng)
        query_pixels.append(pixels)
        query_labels.extend([label] * len(pixels))
    query_pixels = np.concatenate(query_pixels)
    query_labels = np.asarray(query_labels, dtype=np.int8)
    queries = np.column_stack((np.zeros(len(query_pixels)), query_pixels))

    video = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2)[None].float().cuda()
    queries = torch.as_tensor(queries[None], dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        model(video_chunk=video[:, :model.step * 2], is_first_step=True, queries=queries)
        for start in range(0, frame_count - model.step, model.step):
            tracks, predicted_visible = model(
                video_chunk=video[:, start:start + model.step * 2]
            )
    tracks = tracks[0].cpu().numpy()
    predicted_visible = predicted_visible[0].cpu().numpy().astype(bool)
    del video

    track_dir = output / "cotracker"
    track_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        track_dir / f"{camera}.npz", tracks=tracks,
        visibility=predicted_visible, labels=query_labels,
    )

    k = np.asarray(metadata["intrinsics"][camera])
    points = np.zeros((frame_count, len(query_labels), 3), dtype=np.float32)
    colors = np.zeros_like(points)
    visible = np.zeros((frame_count, len(query_labels)), dtype=bool)
    mask_counts = []
    for frame in range(frame_count):
        frame_masks = segment(rgb_frames[frame])
        depth = np.load(source / "depth" / str(camera) / f"{frame}.npy")
        valid_depth = (depth > 200) & (depth < 1500)
        frame_masks &= valid_depth[None]
        mask_counts.append(frame_masks.reshape(3, -1).sum(axis=1).tolist())
        for label in range(3):
            mask_path = output / "mask" / str(camera) / str(label) / f"{frame}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame_masks[label].astype(np.uint8) * 255).save(mask_path)

        x_float, y_float = tracks[frame, :, 0], tracks[frame, :, 1]
        x = np.clip(np.rint(x_float).astype(int), 0, width - 1)
        y = np.clip(np.rint(y_float).astype(int), 0, height - 1)
        in_frame = (x_float >= 0) & (x_float < width) & (y_float >= 0) & (y_float < height)
        visible[frame] = (
            predicted_visible[frame] & in_frame & frame_masks[query_labels, y, x]
        )

        d = depth[y, x] / 1000.0
        # Match pcd_utils.getPcdFromDepth and the exporter's adjusted intrinsics.
        camera_points = np.column_stack((
            (k[0, 2] - (width - x_float)) * d / k[0, 0],
            (y_float - k[1, 2]) * d / k[1, 1],
            d,
        ))
        points[frame] = (camera_points @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
        colors[frame] = rgb_frames[frame, y, x].astype(np.float32) / 255.0

    return query_labels, points, colors, visible, np.asarray(mask_counts)


def process(source, output):
    metadata = json.loads((source / "metadata.json").read_text())
    with (source / "calibrate.pkl").open("rb") as stream:
        c2ws = np.asarray(pickle.load(stream))
    frame_count = metadata["frame_num"]
    output.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source / "color", output / "color", dirs_exist_ok=True)
    shutil.copytree(source / "depth", output / "depth", dirs_exist_ok=True)
    shutil.copy2(source / "metadata.json", output / "metadata.json")
    shutil.copy2(source / "calibrate.pkl", output / "calibrate.pkl")

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.cuda().eval()
    rng = np.random.default_rng(42)
    by_class = {label: {"points": [], "colors": [], "visible": []} for label in range(3)}
    mask_counts = {}
    for camera in range(len(c2ws)):
        print(f"Tracking camera {camera}", flush=True)
        labels, points, colors, visible, counts = track_camera(
            model, source, output, camera, metadata, c2ws[camera], rng
        )
        mask_counts[str(camera)] = counts.tolist()
        for label in range(3):
            selected = labels == label
            by_class[label]["points"].append(points[:, selected])
            by_class[label]["colors"].append(colors[:, selected])
            by_class[label]["visible"].append(visible[:, selected])

    for label in range(3):
        for key in ("points", "colors", "visible"):
            by_class[label][key] = np.concatenate(by_class[label][key], axis=1)

    object_points = by_class[0]["points"]
    object_colors = by_class[0]["colors"]
    object_visible = by_class[0]["visible"]
    initial_pcd = o3d.geometry.PointCloud()
    initial_pcd.points = o3d.utility.Vector3dVector(object_points[0])
    clusters = np.asarray(initial_pcd.cluster_dbscan(eps=0.015, min_points=5))
    largest_cluster = np.bincount(clusters[clusters >= 0]).argmax()
    displacement = np.linalg.norm(object_points - object_points[0], axis=-1)
    largest_observed_motion = np.where(object_visible, displacement, 0).max(axis=0)
    keep = (
        object_visible[0] & (object_visible.sum(axis=0) >= 10)
        & (clusters == largest_cluster) & (largest_observed_motion < 0.05)
    )
    object_points = object_points[:, keep]
    object_colors = object_colors[:, keep]
    object_visible = object_visible[:, keep]
    object_motion_valid = np.zeros_like(object_visible)
    displacement = np.linalg.norm(object_points[1:] - object_points[:-1], axis=-1)
    object_motion_valid[:-1] = (
        object_visible[:-1] & object_visible[1:] & (displacement < 0.05)
    )
    object_points = fill_missing(object_points, object_visible)
    object_colors = fill_missing(object_colors, object_visible)

    controller_parts = []
    controller_labels = []
    for label in (1, 2):
        points = by_class[label]["points"]
        visible = by_class[label]["visible"]
        score = visible.sum(axis=0)
        selected = np.argsort(-score)[:CONTROLLER_POINTS_PER_GRIPPER]
        controller_parts.append(fill_missing(points[:, selected], visible[:, selected]))
        controller_labels.extend([label - 1] * len(selected))
    controller_points = np.concatenate(controller_parts, axis=1)
    controller_labels = np.asarray(controller_labels, dtype=np.int64)

    final_data = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": object_visible,
        "object_motions_valid": object_motion_valid,
        "controller_points": controller_points,
        "controller_labels": controller_labels,
        "controller_mask": np.ones(len(controller_labels), dtype=bool),
        "surface_points": np.zeros((0, 3), dtype=np.float32),
        "interior_points": np.zeros((0, 3), dtype=np.float32),
    }
    with (output / "final_data.pkl").open("wb") as stream:
        pickle.dump(final_data, stream)
    train_end = frame_count - 6
    split = {
        "frame_len": frame_count,
        "train": [0, train_end],
        "test": [train_end, frame_count],
    }
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    summary = {
        "source": str(source),
        "mask_names": MASK_NAMES,
        "object_tracks": int(object_points.shape[1]),
        "object_visible_per_frame": object_visible.sum(axis=1).tolist(),
        "object_motion_valid_per_frame": object_motion_valid.sum(axis=1).tolist(),
        "controller_tracks_per_gripper": [len(part[0]) for part in controller_parts],
        "mask_pixel_counts": mask_counts,
    }
    (output / "processing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text(
        "# 奶龙合成数据：EMPM 离线参数估计输入\n\n"
        f"原始 RGB-D 数据：`{source}`。\n\n"
        "`final_data.pkl` 包含物体的 3D 点轨迹、可见性、运动有效标记和 30 条夹爪点轨迹。"
        "`controller_labels` 把控制点分给两只夹爪：0 为橙色，1 为蓝色，各 15 点。"
        f"`split.json` 将 {frame_count} 帧分为训练 {train_end} 帧、测试 6 帧，"
        "使训练段覆盖主要夹爪拉动过程。"
        "`mask/` 和 `cotracker/` 保存中间结果；RGB、深度、内外参也已复制到本目录。\n\n"
        "HSV 掩码只适用于这段合成画面的配色。头顶棕色圆环只分割出一部分。"
        "`surface_points` 和 `interior_points` 为空；EMPM 初始粒子来自 RGB-D 跟踪到的表面点。\n\n"
        "在 EMPM 仓库根目录运行离线参数估计：\n\n"
        f"```bash\n.conda/empm/bin/python scripts/train_test/train_empm.py --case-dir {output}\n```\n"
    )
    print(f"Saved {output / 'final_data.pkl'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    process(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
