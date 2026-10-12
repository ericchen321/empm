"""Transfer the existing tracked-particle identities through all saved FEM states.

This supplies a synthetic rollout diagnostic, not new RGB-D tracking or fitting.
The original source-step-80 points are retained exactly as the reference.
"""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


GRIPPER_LINKS = ((8, 9, 10), (20, 21, 22))


def pose_rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[..., [4, 5, 6, 3]]).as_matrix()


def load_recording(source):
    export = source / "blender_export"
    scene = json.loads((export / "scene.json").read_text())
    recording = json.loads((export / "recording.json").read_text())
    arrays = [np.load(export / row["path"]) for row in recording["chunks"]]
    steps = np.concatenate([a["control_step"] for a in arrays])
    poses = np.concatenate([a["rigid_pose"] for a in arrays])
    times = np.concatenate([a["time_s"] for a in arrays])
    keep = steps >= 0
    return scene, steps[keep], times[keep], poses[keep]


def rigid_face_transfer(reference, original_points, current):
    """Move each point with its reference triangle's least-squares rigid motion."""
    a = reference - reference.mean(axis=1, keepdims=True)
    b = current - current.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("nki,nkj->nij", a, b))
    reflected = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)) < 0
    vt[reflected, -1, :] *= -1
    rotation = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
    offset = original_points - reference.mean(axis=1)
    return np.einsum("nij,nj->ni", rotation, offset) + current.mean(axis=1)


def rigid_face_velocity(triangles, velocities, points):
    """Fit instantaneous triangle translation/rotation using saved FEM velocities."""
    r = triangles - triangles.mean(axis=1, keepdims=True)
    v = velocities - velocities.mean(axis=1, keepdims=True)
    inertia = (np.sum(r * r, axis=(1, 2))[:, None, None] * np.eye(3)
               - np.einsum("nki,nkj->nij", r, r))
    torque = np.cross(r, v).sum(axis=1)
    omega = np.linalg.solve(inertia, torque[..., None])[..., 0]
    return velocities.mean(axis=1) + np.cross(omega, points - triangles.mean(axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6"))
    parser.add_argument("--case-dir", type=Path,
                        default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--field", type=Path,
                        default=Path("outputs/nailoong_material_init/field.npz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/nailoong_full_sequence"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.case_dir / "final_data.pkl").open("rb") as stream:
        old = pickle.load(stream)
    field = np.load(args.field)
    reference_points = old["object_points"][0]
    reference_controllers = old["controller_points"][0]
    reference_step = json.loads((args.case_dir / "metadata.json").read_text())["source_steps"][0]
    scene, steps, times, poses = load_recording(args.source)
    states = [np.load(args.source / "states" / f"{int(step):06d}.npz") for step in steps]
    vertices = np.stack([s["positions_m"] for s in states])
    reference_index = int(np.flatnonzero(steps == reference_step)[0])
    faces = field["nearest_source_face_vertex_ids"]
    reference_faces = vertices[reference_index, faces]
    points = np.stack([rigid_face_transfer(reference_faces, reference_points, v[faces])
                       for v in vertices]).astype(np.float32)
    velocities = np.stack([
        rigid_face_velocity(v[faces], state["velocities_mps"][faces], p)
        for v, state, p in zip(vertices, states, points)
    ])

    geometry = np.load(args.source / "blender_export/geometry.npz")
    link_binding = np.zeros(len(reference_controllers), dtype=np.int64)
    local_controller = np.zeros_like(reference_controllers)
    distances = np.zeros(len(reference_controllers))
    for role, links in enumerate(GRIPPER_LINKS):
        selected = np.flatnonzero(old["controller_labels"] == role)
        distance_per_link = []
        for link in links:
            mesh = next(o for o in scene["objects"] if o.get("link_id") == f"link_{link:04d}")
            local = geometry[mesh["vertices_key"]]
            mesh_pose = np.asarray(mesh["mesh_to_link"])
            local = local @ pose_rotation(mesh_pose).T + mesh_pose[:3]
            pose = poses[reference_index, link]
            world = local @ pose_rotation(pose).T + pose[:3]
            distance_per_link.append(cKDTree(world).query(reference_controllers[selected])[0])
        distance_per_link = np.stack(distance_per_link, axis=1)
        nearest = distance_per_link.argmin(axis=1)
        link_binding[selected] = np.asarray(links)[nearest]
        distances[selected] = distance_per_link[np.arange(len(selected)), nearest]
        for point in selected:
            pose = poses[reference_index, link_binding[point]]
            local_controller[point] = (reference_controllers[point] - pose[:3]) @ pose_rotation(pose)
    controllers = []
    for frame_poses in poses:
        p = frame_poses[link_binding]
        controllers.append(np.einsum("nij,nj->ni", pose_rotation(p), local_controller) + p[:, :3])
    controllers = np.asarray(controllers, dtype=np.float32)
    np.savez_compressed(
        args.output_dir / "source_tracks.npz",
        object_points_world_m=points, object_velocities_world_mps=velocities.astype(np.float32),
        controller_points_world_m=controllers,
        object_visibilities=np.ones(points.shape[:2], dtype=bool),
        controller_labels=old["controller_labels"], controller_rigid_links=link_binding,
        part_labels=field["part_labels"], source_steps=steps, time_s=times,
        rigid_poses=poses, fps=np.asarray(scene["fps"]),
    )
    displacement = np.linalg.norm(points - points[0], axis=-1)
    description = {
        "source_run": str(args.source), "frame_count": len(steps), "fps": scene["fps"],
        "time_s": [float(times[0]), float(times[-1])], "reference_source_step": reference_step,
        "particle_mapping": "Existing 1765 RGB-D particles moved by nearest source triangle local rigid transform; exact at source step 80.",
        "velocity_mapping": "Instantaneous rigid triangle velocity fitted from saved FEM vertex velocities at that same source state; no future frames used.",
        "controller_mapping": "Each existing controller point bound to nearest recorded rigid gripper link at source step 80; fingers articulate independently.",
        "visibility": "All source-mapped points, including hidden points; not the earlier RGB-D visible-point metric.",
        "reference_particle_max_difference_m": float(np.abs(points[reference_index] - reference_points).max()),
        "reference_controller_max_difference_m": float(np.abs(controllers[reference_index] - reference_controllers).max()),
        "controller_surface_distance_max_m": float(distances.max()),
        "maximum_point_displacement_m": float(displacement.max()),
        "rms_displacement_source0_to80_m": float(np.sqrt(np.mean(displacement[reference_index] ** 2))),
        "initial_z_min_m": float(points[0, :, 2].min()),
    }
    (args.output_dir / "source_mapping.json").write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps(description, indent=2))


if __name__ == "__main__":
    main()
