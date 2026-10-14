"""Recorded Nailoong finger attachments for selected material tet particles."""

import json

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation, Slerp

from phys_sim.diff_simulator.warp_mpm.mpm_data_structure import MPMStateStruct


PATCH_ROLES = (("belly", 0), ("belly", 0), ("ring", 2), ("ring", 2))
WORLD_TO_SIM = 5.0


@wp.kernel
def _before_patch_step(
    state: MPMStateStruct,
    full_ids: wp.array(dtype=wp.int32),
    patch_ids: wp.array(dtype=wp.int32),
    activation_ticks: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    position_secants: wp.array(dtype=wp.vec3),
    rotation_secants: wp.array(dtype=wp.mat33),
    offsets: wp.array(dtype=wp.vec3),
    captures: wp.array(dtype=wp.vec3),
    capture_targets: wp.array(dtype=wp.vec3),
    endpoint_velocities: wp.array(dtype=wp.vec3),
    tick: int,
    frame: int,
    patch_count: int,
    particle_count: int,
):
    i = wp.tid()
    patch = patch_ids[i]
    if tick < activation_ticks[patch]:
        return
    particle = full_ids[i]
    table_index = tick * patch_count + patch
    if tick == activation_ticks[patch]:
        capture = state.particle_x[particle]
        offset = wp.transpose(rotations[table_index]) * (capture - positions[table_index])
        offsets[i] = offset
        captures[i] = capture
        capture_targets[i] = positions[table_index] + rotations[table_index] * offset
        endpoint_velocities[frame * particle_count + i] = state.particle_v[particle]
    state.particle_v[particle] = (position_secants[table_index]
                                  + rotation_secants[table_index] * offsets[i])


@wp.kernel
def _after_patch_step(
    state: MPMStateStruct,
    full_ids: wp.array(dtype=wp.int32),
    patch_ids: wp.array(dtype=wp.int32),
    activation_ticks: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    position_secants: wp.array(dtype=wp.vec3),
    rotation_secants: wp.array(dtype=wp.mat33),
    offsets: wp.array(dtype=wp.vec3),
    endpoint_velocities: wp.array(dtype=wp.vec3),
    tick: int,
    frame: int,
    last_substep: int,
    patch_count: int,
    particle_count: int,
):
    i = wp.tid()
    patch = patch_ids[i]
    if tick < activation_ticks[patch]:
        return
    particle = full_ids[i]
    table_index = tick * patch_count + patch
    target_index = (tick + 1) * patch_count + patch
    prescribed_v = (position_secants[table_index]
                    + rotation_secants[table_index] * offsets[i])
    state.particle_x[particle] = positions[target_index] + rotations[target_index] * offsets[i]
    state.particle_v[particle] = prescribed_v
    if last_substep != 0:
        endpoint_velocities[(frame + 1) * particle_count + i] = state.particle_v[particle]


def _rotation_from_wxyz(poses):
    return Rotation.from_quat(poses[..., [4, 5, 6, 3]])


def prepare_recorded_finger_patches(source_run, source_steps, rigid_poses,
                                    observation_count, shift, frame_dt, sim_dt, device):
    """Prepare fixed masks and substep pose tables once for an entire fit/replay."""
    source_steps = np.asarray(source_steps)
    if int(source_steps[0]) > 39:
        raise ValueError("recorded_finger_patches requires a start at or before source state 39")
    with np.load(source_run / "initial_state.npz") as initial:
        tets = initial["tetrahedra"]
        tet_labels = initial["tet_part_labels"]
    binding_path = source_run / "snap_bindings.json"
    bindings = json.loads(binding_path.read_text())
    patches = []
    tet_ids = []
    patch_ids = []
    for patch_id, (role, part_label) in enumerate(PATCH_ROLES):
        binding = bindings[role]["patches"][patch_id % 2]
        vertices = binding["vertex_ids"]
        selected = np.flatnonzero((tet_labels == part_label) &
                                  np.isin(tets, vertices).any(axis=1)).astype(np.int32)
        tet_ids.extend(selected.tolist())
        patch_ids.extend([patch_id] * len(selected))
        patches.append({
            "patch_id": patch_id, "role": role,
            "part": "body" if part_label == 0 else "ring",
            "part_label": part_label, "link": int(binding["link_idx"]),
            "activation_source_step": int(bindings[role]["step"]) - 1,
            "source_vertex_ids": vertices,
            "source_local_anchors_m": binding["local_anchors_m"],
            "source_capture_positions_m": binding["capture_positions_m"],
            "tet_count": len(selected), "tet_ids": selected.tolist(),
        })
    tet_ids = np.asarray(tet_ids, dtype=np.int32)
    patch_ids = np.asarray(patch_ids, dtype=np.int32)
    assert [p["tet_count"] for p in patches] == [34, 40, 47, 7]
    assert len(np.unique(tet_ids)) == len(tet_ids) == 128
    links = np.asarray([p["link"] for p in patches], dtype=np.int32)
    poses = np.ascontiguousarray(rigid_poses[:, links].astype(np.float64))
    frame_count = len(source_steps)
    n_substeps = round(frame_dt / sim_dt)
    tick_count = (frame_count - 1) * n_substeps + 1
    local_times = np.arange(tick_count, dtype=np.float64) / n_substeps
    frame_index = np.minimum(np.arange(tick_count) // n_substeps, frame_count - 2)
    fraction = local_times - frame_index
    p_world = (poses[frame_index, :, :3] * (1.0 - fraction[:, None, None])
               + poses[frame_index + 1, :, :3] * fraction[:, None, None])
    rotations = np.empty((tick_count, len(patches), 3, 3), dtype=np.float64)
    for patch_id in range(len(patches)):
        slerp = Slerp(np.arange(frame_count, dtype=np.float64),
                      _rotation_from_wxyz(poses[:, patch_id]))
        rotations[:, patch_id] = slerp(local_times).as_matrix()
    p_sim = .5 + p_world / WORLD_TO_SIM - np.asarray(shift)
    dp_sim_dt = np.diff(p_world, axis=0) / (WORLD_TO_SIM * sim_dt)
    dR_dt = np.diff(rotations, axis=0) / sim_dt
    activation_steps = np.asarray([p["activation_source_step"] for p in patches],
                                  dtype=np.int32)
    activation_ticks = (activation_steps - int(source_steps[0])) * n_substeps
    flat_vec = lambda values: wp.array(
        np.ascontiguousarray(values.reshape(-1, 3).astype(np.float32)),
        dtype=wp.vec3, device=device)
    flat_mat = lambda values: wp.array(
        np.ascontiguousarray(values.reshape(-1, 3, 3).astype(np.float32)),
        dtype=wp.mat33, device=device)
    return {
        "source_steps": source_steps.copy(), "poses_xyz_wxyz": poses.astype(np.float32),
        "patches": patches, "tet_ids": tet_ids,
        "full_ids": observation_count + tet_ids,
        "patch_ids": patch_ids, "links": links,
        "part_labels": np.asarray([p["part_label"] for p in patches], dtype=np.int32),
        "activation_steps": activation_steps,
        "shift": np.asarray(shift, dtype=np.float64), "sim_dt": sim_dt,
        "n_substeps": n_substeps, "device": device,
        "full_ids_device": wp.array(observation_count + tet_ids, dtype=wp.int32, device=device),
        "patch_ids_device": wp.array(patch_ids, dtype=wp.int32, device=device),
        "activation_ticks_device": wp.array(activation_ticks, dtype=wp.int32, device=device),
        "positions_device": flat_vec(p_sim),
        "rotations_device": flat_mat(rotations),
        "position_secants_device": flat_vec(dp_sim_dt),
        "rotation_secants_device": flat_mat(dR_dt),
    }


class RecordedFingerMaterialPatches:
    """Fresh capture offsets and endpoint velocities for one simulated field."""

    def __init__(self, prepared):
        self.prepared = prepared
        device = prepared["device"]
        count = len(prepared["full_ids"])
        self.offsets = wp.zeros(count, dtype=wp.vec3, device=device)
        self.captures = wp.zeros(count, dtype=wp.vec3, device=device)
        self.capture_targets = wp.zeros(count, dtype=wp.vec3, device=device)
        self.endpoint_velocities = wp.zeros(len(prepared["source_steps"]) * count,
                                            dtype=wp.vec3, device=device)

    def before_substep(self, prev_state, frame, substep):
        prepared = self.prepared
        tick = frame * prepared["n_substeps"] + substep
        if tick < (39 - int(prepared["source_steps"][0])) * prepared["n_substeps"]:
            return
        wp.launch(_before_patch_step, dim=len(prepared["full_ids"]), inputs=[
            prev_state, prepared["full_ids_device"], prepared["patch_ids_device"],
            prepared["activation_ticks_device"], prepared["positions_device"],
            prepared["rotations_device"], prepared["position_secants_device"],
            prepared["rotation_secants_device"], self.offsets, self.captures,
            self.capture_targets, self.endpoint_velocities, tick, frame, 4,
            len(prepared["full_ids"])], device=prepared["device"])

    def after_substep(self, next_state, frame, substep):
        prepared = self.prepared
        tick = frame * prepared["n_substeps"] + substep
        if tick < (39 - int(prepared["source_steps"][0])) * prepared["n_substeps"]:
            return
        wp.launch(_after_patch_step, dim=len(prepared["full_ids"]), inputs=[
            next_state, prepared["full_ids_device"], prepared["patch_ids_device"],
            prepared["activation_ticks_device"], prepared["positions_device"],
            prepared["rotations_device"], prepared["position_secants_device"],
            prepared["rotation_secants_device"], self.offsets,
            self.endpoint_velocities, tick, frame,
            int(substep == prepared["n_substeps"] - 1), 4,
            len(prepared["full_ids"])], device=prepared["device"])

    def diagnostics(self, full_prediction):
        """Read small GPU buffers once; reconstruct frame targets from recorded poses."""
        prepared = self.prepared
        count = len(prepared["full_ids"])
        steps = prepared["source_steps"]
        poses = prepared["poses_xyz_wxyz"].astype(np.float64)
        patch_ids = prepared["patch_ids"]
        capture_steps = prepared["activation_steps"][patch_ids]
        shift = prepared["shift"]
        to_world = lambda values: (values + shift - .5) * WORLD_TO_SIM
        captured_world = to_world(self.captures.numpy().astype(np.float64))
        capture_target_world = to_world(self.capture_targets.numpy().astype(np.float64))
        offsets_world = self.offsets.numpy().astype(np.float64) * WORLD_TO_SIM
        actual_velocity = self.endpoint_velocities.numpy().reshape(len(steps), count, 3)
        actual_velocity = actual_velocity.astype(np.float64) * WORLD_TO_SIM
        actual_position = full_prediction[:, prepared["full_ids"]]
        target_position = np.full_like(actual_position, np.nan, dtype=np.float64)
        expected_velocity = np.full_like(target_position, np.nan)
        active = steps[:, None] >= capture_steps[None, :]
        actual_velocity[~active] = np.nan
        capture_frame_displacement = []
        capture_reconstruction_displacement = []
        for patch_id, patch in enumerate(prepared["patches"]):
            selected = np.flatnonzero(patch_ids == patch_id)
            capture_frame = int(np.flatnonzero(steps == patch["activation_source_step"])[0])
            capture_pose = poses[capture_frame, patch_id]
            capture_rotation = _rotation_from_wxyz(capture_pose).as_matrix()
            r_cpu = (captured_world[selected] - capture_pose[:3]) @ capture_rotation
            capture_reconstruction_displacement.extend(np.linalg.norm(
                r_cpu - offsets_world[selected], axis=1) * 1000.0)
            capture_frame_displacement.extend(np.linalg.norm(
                actual_position[capture_frame, selected] - captured_world[selected],
                axis=1) * 1000.0)
            for frame in range(capture_frame, len(steps)):
                pose = poses[frame, patch_id]
                rotation = _rotation_from_wxyz(pose).as_matrix()
                target_position[frame, selected] = pose[:3] + r_cpu @ rotation.T
                if frame > capture_frame:
                    previous_pose = poses[frame - 1, patch_id]
                    before_last_tick_p = (previous_pose[:3] / prepared["n_substeps"]
                                          + pose[:3] * (1.0 - 1.0 / prepared["n_substeps"]))
                    interpolation = Slerp([0.0, 1.0], _rotation_from_wxyz(
                        np.stack((previous_pose, pose))))
                    before_last_tick_R = interpolation(
                        1.0 - 1.0 / prepared["n_substeps"]).as_matrix()
                    previous_target = before_last_tick_p + r_cpu @ before_last_tick_R.T
                    expected_velocity[frame, selected] = (
                        target_position[frame, selected] - previous_target) / prepared["sim_dt"]
        position_error = np.linalg.norm(actual_position[active] - target_position[active], axis=1)
        capture_displacement = np.linalg.norm(captured_world - capture_target_world, axis=1)
        velocity_mask = np.isfinite(expected_velocity[..., 0])
        velocity_error = np.linalg.norm(
            actual_velocity[velocity_mask] - expected_velocity[velocity_mask], axis=1)
        return {
            "capture_world_m": captured_world.astype(np.float32),
            "capture_target_world_m": capture_target_world.astype(np.float32),
            "capture_source_step": capture_steps,
            "finger_local_offset_world_m": offsets_world.astype(np.float32),
            "actual_endpoint_world_m": actual_position,
            "target_endpoint_world_m": target_position.astype(np.float32),
            "active_endpoint_mask": active,
            "actual_endpoint_velocity_world_mps": actual_velocity.astype(np.float32),
            "expected_endpoint_velocity_world_mps": expected_velocity.astype(np.float32),
            "summary": {
                "max_active_endpoint_position_error_mm": float(position_error.max() * 1000.0),
                "max_capture_reconstruction_displacement_mm": float(
                    np.max(capture_reconstruction_displacement)),
                "max_capture_target_displacement_mm": float(capture_displacement.max() * 1000.0),
                "max_capture_vs_returned_frame_position_discrepancy_mm": float(
                    np.max(capture_frame_displacement)),
                "max_active_endpoint_velocity_error_mps": float(velocity_error.max()),
            },
        }
