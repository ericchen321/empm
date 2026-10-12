"""Sample HAG4R gripper meshes as rigid, oriented Gaussian surface splats."""

import json
from pathlib import Path

import numpy as np


GRIPPER_MESHES = (8, 9, 10, 19, 20, 21)
TABLE_MESH = 22


def _quat_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw), axis=-1)


def _rotate(quat, vectors):
    qvec = quat[..., 1:]
    cross = 2 * np.cross(qvec, vectors)
    return vectors + quat[..., :1] * cross + np.cross(qvec, cross)


def _normal_quaternions(normals):
    """Rotate the Gaussian's thin local Z axis onto each triangle normal."""
    quats = np.column_stack((1 + normals[:, 2], -normals[:, 1],
                             normals[:, 0], np.zeros(len(normals), dtype=np.float32)))
    opposite = quats[:, 0] < 1e-5
    quats[opposite] = (0, 1, 0, 0)
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def _sample_surface(vertices, faces, spacing, rng, top_only=False):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    if top_only:
        areas[cross[:, 2] <= 0] = 0
    cumulative = np.cumsum(areas, dtype=np.float64)
    count = max(1, round(float(cumulative[-1]) / spacing**2))
    selected = np.searchsorted(cumulative, rng.random(count) * cumulative[-1])
    triangle = triangles[selected]
    bary_u = np.sqrt(rng.random(count))
    bary_v = rng.random(count)
    points = ((1 - bary_u)[:, None] * triangle[:, 0]
              + (bary_u * (1 - bary_v))[:, None] * triangle[:, 1]
              + (bary_u * bary_v)[:, None] * triangle[:, 2])
    normals = cross[selected] / (2 * areas[selected, None])
    return points.astype(np.float32), normals.astype(np.float32)


def build_rigid_assets(source_dir, include_table=True):
    """Return one sampled splat asset in link-local meters and wxyz quaternions.

    `source_dir` points to `run_v6` or its `blender_export`. `link_indices` are source
    `rigid_poses` indices for the grippers; -1 denotes the static table, whose
    link-local points are transformed with `static_pose` by pose_rigid_assets.
    """
    source_dir = Path(source_dir)
    if not (source_dir / "scene.json").exists():
        source_dir = source_dir / "blender_export"
    scene = json.loads((source_dir / "scene.json").read_text())
    objects = {int(item["id"].split("_")[-1]): item for item in scene["objects"]}
    links = {item["id"]: item for item in scene["rigid_links"]}
    materials = {item["id"]: item for item in scene["materials"]}
    geometry = np.load(source_dir / scene["geometry_file"])
    rng = np.random.default_rng(20261012)
    fields = {key: [] for key in ("local_xyz", "local_quaternions", "local_normals",
                                   "colors", "scales", "opacity", "link_indices")}
    table_pose = None
    for index in (*GRIPPER_MESHES, *((TABLE_MESH,) if include_table else ())):
        mesh = objects[index]
        link = links[mesh["link_id"]]
        is_table = index == TABLE_MESH
        spacing = .006 if is_table else .0012
        points, normals = _sample_surface(geometry[mesh["vertices_key"]],
                                          geometry[mesh["faces_key"]], spacing, rng,
                                          top_only=is_table)
        mesh_pose = np.asarray(mesh["mesh_to_link"], dtype=np.float32)
        local_quats = _quat_multiply(mesh_pose[3:], _normal_quaternions(normals))
        points = _rotate(mesh_pose[3:], points) + mesh_pose[:3]
        normals = _rotate(mesh_pose[3:], normals)
        count = len(points)
        color = np.asarray(materials[mesh["material_id"]]["base_color_rgba"][:3],
                           dtype=np.float32)
        fields["local_xyz"].append(points)
        fields["local_quaternions"].append(local_quats.astype(np.float32))
        fields["local_normals"].append(normals.astype(np.float32))
        fields["colors"].append(np.repeat(color[None], count, axis=0))
        radius = spacing * (1.1 if is_table else .72)
        fields["scales"].append(np.tile((radius, radius, .00018 if is_table else .00012),
                                        (count, 1)).astype(np.float32))
        fields["opacity"].append(np.full((count, 1), .99 if is_table else .95, dtype=np.float32))
        fields["link_indices"].append(np.full(count, -1 if is_table else link["record_index"],
                                               dtype=np.int16))
        if is_table:
            table_pose = np.asarray(link["initial_pose"], dtype=np.float32)
    asset = {key: np.concatenate(chunks) for key, chunks in fields.items()}
    asset["static_pose"] = table_pose
    return asset


def pose_rigid_assets(asset, rigid_poses):
    """Apply one source frame's `(24, 7)` xyz+wxyz poses; return xyz, wxyz, RGB."""
    rigid_poses = np.asarray(rigid_poses, dtype=np.float32)
    worldxyz = np.empty_like(asset["local_xyz"])
    worldquats = np.empty_like(asset["local_quaternions"])
    worldnormals = np.empty_like(asset["local_normals"])
    for index in np.unique(asset["link_indices"]):
        select = asset["link_indices"] == index
        pose = asset["static_pose"] if index == -1 else rigid_poses[index]
        worldxyz[select] = _rotate(pose[3:], asset["local_xyz"][select]) + pose[:3]
        worldquats[select] = _quat_multiply(pose[3:], asset["local_quaternions"][select])
        worldnormals[select] = _rotate(pose[3:], asset["local_normals"][select])
    light = np.asarray((-0.35, -0.4, 0.85), dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.maximum(worldnormals @ light, 0)
    rgb = np.clip(asset["colors"] * (.82 + .23 * diffuse[:, None]), 0, 1)
    return worldxyz, worldquats, rgb
