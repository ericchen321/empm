"""Four recorded finger centroids coupled through the ordinary MPM grid."""

import numpy as np
import warp as wp

from phys_sim.diff_simulator.warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct
from scripts.eval.nailoong_recorded_finger_patches import WORLD_TO_SIM, _rotation_from_wxyz


@wp.struct
class _Work:
    weights: wp.array(dtype=wp.float64)
    moments: wp.array(dtype=wp.float64)
    masses: wp.array(dtype=wp.float64)
    offsets: wp.array(dtype=wp.float64)
    captures: wp.array(dtype=wp.float64)
    capture_targets: wp.array(dtype=wp.float64)
    captured: wp.array(dtype=wp.int32)
    A: wp.array(dtype=wp.float64)
    B: wp.array(dtype=wp.float64)
    pre: wp.array(dtype=wp.float64)
    post: wp.array(dtype=wp.float64)
    rhs: wp.array(dtype=wp.float64)
    velocity: wp.array(dtype=wp.float64)
    centroid: wp.array(dtype=wp.float64)
    target: wp.array(dtype=wp.float64)
    multiplier: wp.array(dtype=wp.float64)
    residual: wp.array(dtype=wp.float64)
    failure: wp.array(dtype=wp.int32)
    failed_tick: wp.array(dtype=wp.int32)
    max_residual: wp.array(dtype=wp.float64)
    max_tick: wp.array(dtype=wp.int32)
    max_row: wp.array(dtype=wp.int32)


@wp.struct
class _Snapshots:
    tick: wp.array(dtype=wp.int32)
    active: wp.array(dtype=wp.int32)
    A: wp.array(dtype=wp.float64)
    B: wp.array(dtype=wp.float64)
    rhs: wp.array(dtype=wp.float64)
    velocity: wp.array(dtype=wp.float64)
    centroid: wp.array(dtype=wp.float64)
    target: wp.array(dtype=wp.float64)
    multiplier: wp.array(dtype=wp.float64)
    pre: wp.array(dtype=wp.float64)
    post: wp.array(dtype=wp.float64)
    residual: wp.array(dtype=wp.float64)


@wp.kernel
def _build_weights(state: MPMStateStruct, model: MPMModelStruct,
                   full_ids: wp.array(dtype=wp.int32),
                   patch_ids: wp.array(dtype=wp.int32),
                   activation: wp.array(dtype=wp.int32), work: _Work,
                   tick: int, node_count: int):
    selected = wp.tid()
    patch = patch_ids[selected]
    if tick < activation[patch]:
        return
    particle = full_ids[selected]
    mass = wp.float64(state.particle_mass[particle])
    x = state.particle_x[particle]
    for axis in range(3):
        wp.atomic_add(work.moments, patch * 3 + axis, mass * wp.float64(x[axis]))

    grid_pos = x * model.inv_dx
    base_x = wp.int(grid_pos[0] - 0.5)
    base_y = wp.int(grid_pos[1] - 0.5)
    base_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(wp.cw_mul(wa, wa) * 0.5,
                 wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                 wp.cw_mul(wc, wc) * 0.5)
    share = mass / work.masses[patch]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]
                node = ((base_x + i) * model.grid_dim_y + base_y + j) * model.grid_dim_z + base_z + k
                wp.atomic_add(work.weights, patch * node_count + node,
                              share * wp.float64(weight))


@wp.kernel
def _gather_grid(state: MPMStateStruct, model: MPMModelStruct, work: _Work,
                 floor_z: float, node_count: int):
    node = wp.tid()
    z = node % model.grid_dim_z
    y = (node // model.grid_dim_z) % model.grid_dim_y
    x = node // (model.grid_dim_y * model.grid_dim_z)
    w0 = work.weights[node]
    w1 = work.weights[node_count + node]
    w2 = work.weights[2 * node_count + node]
    w3 = work.weights[3 * node_count + node]
    if w0 == 0.0 and w1 == 0.0 and w2 == 0.0 and w3 == 0.0:
        return
    grid_v = state.grid_v_out[x, y, z]
    grid_m = state.grid_m[x, y, z]
    for a in range(4):
        wa = work.weights[a * node_count + node]
        if wa != 0.0:
            for axis in range(3):
                wp.atomic_add(work.pre, a * 3 + axis, wa * wp.float64(grid_v[axis]))
            if grid_m > 1.0e-15 and wp.float(z) * model.dx > floor_z:
                for b in range(a + 1):
                    wb = work.weights[b * node_count + node]
                    if wb != 0.0:
                        wp.atomic_add(work.A, a * 4 + b,
                                      wa * wb / wp.float64(grid_m))


@wp.kernel
def _solve(work: _Work, snapshots: _Snapshots,
           activation: wp.array(dtype=wp.int32),
           positions: wp.array(dtype=wp.vec3),
           rotations: wp.array(dtype=wp.mat33),
           tick: int, dt: float):
    if work.failure[0] != 0:
        return
    valid = True
    for a in range(4):
        if tick >= activation[a]:
            pose = tick * 4 + a
            endpoint = (tick + 1) * 4 + a
            for axis in range(3):
                work.centroid[a * 3 + axis] = work.moments[a * 3 + axis] / work.masses[a]
            if work.captured[a] == 0:
                for axis in range(3):
                    local = wp.float64(0.0)
                    for j in range(3):
                        local += wp.float64(rotations[pose][j, axis]) * (
                            work.centroid[a * 3 + j] - wp.float64(positions[pose][j]))
                    work.offsets[a * 3 + axis] = local
                    work.captures[a * 3 + axis] = work.centroid[a * 3 + axis]
                for axis in range(3):
                    capture_target = wp.float64(positions[pose][axis])
                    for j in range(3):
                        capture_target += (wp.float64(rotations[pose][axis, j]) *
                                           work.offsets[a * 3 + j])
                    work.capture_targets[a * 3 + axis] = capture_target
                work.captured[a] = 1
            for axis in range(3):
                target = wp.float64(positions[endpoint][axis])
                for j in range(3):
                    target += wp.float64(rotations[endpoint][axis, j]) * work.offsets[a * 3 + j]
                work.target[a * 3 + axis] = target
                work.velocity[a * 3 + axis] = (
                    target - work.centroid[a * 3 + axis]) / wp.float64(dt)
                work.rhs[a * 3 + axis] = work.velocity[a * 3 + axis] - work.pre[a * 3 + axis]
                if not wp.isfinite(work.rhs[a * 3 + axis]) or not wp.isfinite(target):
                    valid = False
            if work.A[a * 4 + a] <= 0.0 or not wp.isfinite(work.A[a * 4 + a]):
                valid = False
            for b in range(a):
                work.A[b * 4 + a] = work.A[a * 4 + b]
        else:
            for axis in range(3):
                work.rhs[a * 3 + axis] = wp.float64(0.0)

    for a in range(4):
        for b in range(4):
            value = wp.float64(0.0)
            if tick >= activation[a] and tick >= activation[b]:
                raw = work.A[a * 4 + b]
                if not wp.isfinite(raw):
                    valid = False
                if work.A[a * 4 + a] > 0.0 and work.A[b * 4 + b] > 0.0:
                    da = wp.float64(1.0) / wp.sqrt(work.A[a * 4 + a])
                    db = wp.float64(1.0) / wp.sqrt(work.A[b * 4 + b])
                    value = da * raw * db
            elif a == b:
                value = wp.float64(1.0)
            work.B[a * 4 + b] = value
    B = wp.mat44d(work.B[0], work.B[1], work.B[2], work.B[3],
                  work.B[4], work.B[5], work.B[6], work.B[7],
                  work.B[8], work.B[9], work.B[10], work.B[11],
                  work.B[12], work.B[13], work.B[14], work.B[15])
    determinant = wp.determinant(B)
    if not wp.isfinite(determinant) or determinant == 0.0:
        valid = False
    if not valid:
        work.failure[0] = 1
        work.failed_tick[0] = tick
        return
    inverse = wp.inverse(B)
    for axis in range(3):
        balanced = wp.vec4d(wp.float64(0.0), wp.float64(0.0),
                            wp.float64(0.0), wp.float64(0.0))
        for a in range(4):
            if tick >= activation[a]:
                da = wp.float64(1.0) / wp.sqrt(work.A[a * 4 + a])
                balanced[a] = da * work.rhs[a * 3 + axis]
        solution = inverse * balanced
        for a in range(4):
            if tick >= activation[a]:
                da = wp.float64(1.0) / wp.sqrt(work.A[a * 4 + a])
                work.multiplier[a * 3 + axis] = da * solution[a]
                if not wp.isfinite(work.multiplier[a * 3 + axis]):
                    valid = False
    if not valid:
        work.failure[0] = 1
        work.failed_tick[0] = tick


@wp.kernel
def _apply_and_measure(state: MPMStateStruct, model: MPMModelStruct,
                       work: _Work, floor_z: float, node_count: int):
    if work.failure[0] != 0:
        return
    node = wp.tid()
    z = node % model.grid_dim_z
    y = (node // model.grid_dim_z) % model.grid_dim_y
    x = node // (model.grid_dim_y * model.grid_dim_z)
    mass = state.grid_m[x, y, z]
    if mass > 1.0e-15 and wp.float(z) * model.dx > floor_z:
        old = state.grid_v_out[x, y, z]
        correction_x = wp.float64(0.0)
        correction_y = wp.float64(0.0)
        correction_z = wp.float64(0.0)
        for a in range(4):
            weight = work.weights[a * node_count + node]
            if weight != 0.0:
                correction_x += weight * work.multiplier[a * 3] / wp.float64(mass)
                correction_y += weight * work.multiplier[a * 3 + 1] / wp.float64(mass)
                correction_z += weight * work.multiplier[a * 3 + 2] / wp.float64(mass)
        state.grid_v_out[x, y, z] = old + wp.vec3(
            wp.float32(correction_x), wp.float32(correction_y), wp.float32(correction_z))
    grid_v = state.grid_v_out[x, y, z]
    for a in range(4):
        weight = work.weights[a * node_count + node]
        if weight != 0.0:
            for axis in range(3):
                wp.atomic_add(work.post, a * 3 + axis,
                              weight * wp.float64(grid_v[axis]))


@wp.func
def _snapshot(work: _Work, snapshots: _Snapshots, slot: int, tick: int,
              activation: wp.array(dtype=wp.int32)):
    snapshots.tick[slot] = tick
    for a in range(4):
        snapshots.active[slot * 4 + a] = wp.int32(tick >= activation[a])
        for b in range(4):
            snapshots.A[slot * 16 + a * 4 + b] = work.A[a * 4 + b]
            snapshots.B[slot * 16 + a * 4 + b] = work.B[a * 4 + b]
        for axis in range(3):
            index = slot * 12 + a * 3 + axis
            item = a * 3 + axis
            snapshots.rhs[index] = work.rhs[item]
            snapshots.velocity[index] = work.velocity[item]
            snapshots.centroid[index] = work.centroid[item]
            snapshots.target[index] = work.target[item]
            snapshots.multiplier[index] = work.multiplier[item]
            snapshots.pre[index] = work.pre[item]
            snapshots.post[index] = work.post[item]
            snapshots.residual[index] = work.residual[item]


@wp.kernel
def _finish_step(work: _Work, snapshots: _Snapshots,
                 activation: wp.array(dtype=wp.int32), tick: int,
                 n_substeps: int, failure_slot: int, maximum_slot: int):
    if work.failure[0] != 0:
        if work.failed_tick[0] == tick:
            _snapshot(work, snapshots, failure_slot, tick, activation)
        return
    for a in range(4):
        if tick >= activation[a]:
            for axis in range(3):
                index = a * 3 + axis
                work.residual[index] = work.post[index] - work.velocity[index]
    for a in range(4):
        if tick >= activation[a]:
            squared = wp.float64(0.0)
            for axis in range(3):
                value = work.residual[a * 3 + axis]
                squared += value * value
            norm_world = wp.sqrt(squared) * wp.float64(WORLD_TO_SIM)
            if not wp.isfinite(norm_world):
                work.failure[0] = 2
                work.failed_tick[0] = tick
                _snapshot(work, snapshots, failure_slot, tick, activation)
                return
            if norm_world > work.max_residual[0]:
                work.max_residual[0] = norm_world
                work.max_tick[0] = tick
                work.max_row[0] = a
                _snapshot(work, snapshots, maximum_slot, tick, activation)
    if tick == activation[0]:
        _snapshot(work, snapshots, 0, tick, activation)
    if tick == activation[2]:
        _snapshot(work, snapshots, 1, tick, activation)
    if (tick + 1) % n_substeps == 0:
        _snapshot(work, snapshots, 2 + tick // n_substeps, tick, activation)


class RecordedFingerGridCentroids:
    """Fresh mass-weighted grid projection state for one forward rollout."""

    def __init__(self, prepared, initial_state, model, floor_point):
        self.prepared = prepared
        self.node_count = model.grid_dim_x * model.grid_dim_y * model.grid_dim_z
        self.floor_z = float(floor_point[2])
        self.n_frames = len(prepared["source_steps"])
        self.failure_slot = self.n_frames + 1
        self.maximum_slot = self.n_frames + 2
        self.snapshot_count = self.n_frames + 3
        device = prepared["device"]
        selected_masses = initial_state.particle_mass.numpy()[prepared["full_ids"]].astype(np.float64)
        self.selected_masses = selected_masses
        masses = np.bincount(prepared["patch_ids"], weights=selected_masses,
                             minlength=4).astype(np.float64)
        self.patch_masses = masses
        self.work = _Work()
        w = self.work
        def zeros(count, dtype):
            return wp.zeros(count, dtype=dtype, device=device)
        w.weights = zeros(4 * self.node_count, wp.float64)
        w.moments = zeros(12, wp.float64)
        w.masses = wp.array(masses, dtype=wp.float64, device=device)
        w.offsets = zeros(12, wp.float64)
        w.captures = zeros(12, wp.float64)
        w.capture_targets = zeros(12, wp.float64)
        w.captured = zeros(4, wp.int32)
        for name, count in (("A", 16), ("B", 16), ("pre", 12), ("post", 12),
                            ("rhs", 12), ("velocity", 12), ("centroid", 12),
                            ("target", 12), ("multiplier", 12), ("residual", 12)):
            setattr(w, name, zeros(count, wp.float64))
        w.failure = zeros(1, wp.int32)
        w.failed_tick = wp.array(np.array([-1], dtype=np.int32), device=device)
        w.max_residual = zeros(1, wp.float64)
        w.max_tick = wp.array(np.array([-1], dtype=np.int32), device=device)
        w.max_row = wp.array(np.array([-1], dtype=np.int32), device=device)
        self.snapshots = _Snapshots()
        s = self.snapshots
        s.tick = wp.array(np.full(self.snapshot_count, -1, dtype=np.int32), device=device)
        s.active = zeros(self.snapshot_count * 4, wp.int32)
        for name, width in (("A", 16), ("B", 16), ("rhs", 12), ("velocity", 12),
                            ("centroid", 12), ("target", 12), ("multiplier", 12),
                            ("pre", 12), ("post", 12), ("residual", 12)):
            setattr(s, name, zeros(self.snapshot_count * width, wp.float64))

    def apply(self, state, model, dt, tick):
        p = self.prepared
        if tick < int(p["activation_steps"][0] - p["source_steps"][0]) * p["n_substeps"]:
            return
        w = self.work
        for field in (w.weights, w.moments, w.A, w.B, w.pre, w.post,
                      w.rhs, w.velocity, w.centroid, w.target, w.multiplier, w.residual):
            field.zero_()
        device = p["device"]
        wp.launch(_build_weights, dim=len(p["full_ids"]), inputs=[
            state, model, p["full_ids_device"], p["patch_ids_device"],
            p["activation_ticks_device"], w, tick, self.node_count], device=device)
        wp.launch(_gather_grid, dim=self.node_count, inputs=[
            state, model, w, self.floor_z, self.node_count], device=device)
        wp.launch(_solve, dim=1, inputs=[
            w, self.snapshots, p["activation_ticks_device"], p["positions_device"],
            p["rotations_device"], tick, dt], device=device)
        wp.launch(_apply_and_measure, dim=self.node_count, inputs=[
            state, model, w, self.floor_z, self.node_count], device=device)
        wp.launch(_finish_step, dim=1, inputs=[
            w, self.snapshots, p["activation_ticks_device"], tick,
            p["n_substeps"], self.failure_slot, self.maximum_slot], device=device)

    def finish_frame(self):
        return bool(self.work.failure.numpy()[0])

    def failure_tick(self):
        return int(self.work.failed_tick.numpy()[0])

    def diagnostics(self, full_prediction):
        p = self.prepared
        steps = p["source_steps"][:len(full_prediction)]
        to_world = lambda x: (x + p["shift"] - .5) * WORLD_TO_SIM
        captures = to_world(self.work.captures.numpy().reshape(4, 3))
        capture_targets = to_world(self.work.capture_targets.numpy().reshape(4, 3))
        offsets = self.work.offsets.numpy().reshape(4, 3) * WORLD_TO_SIM
        captured = self.work.captured.numpy().astype(bool)
        actual = np.full((len(steps), 4, 3), np.nan, dtype=np.float64)
        target = np.full_like(actual, np.nan)
        active = np.zeros((len(steps), 4), dtype=bool)
        capture_frame_discrepancy = []
        capture_reconstruction_discrepancy = []
        poses = p["poses_xyz_wxyz"].astype(np.float64)
        for a in range(4):
            selected = np.flatnonzero(p["patch_ids"] == a)
            ids = p["full_ids"][selected]
            masses = self.selected_masses[selected]
            actual[:, a] = np.einsum("fpc,p->fc", full_prediction[:, ids], masses) / masses.sum()
            capture_frame = int(p["activation_steps"][a] - steps[0])
            if not captured[a] or capture_frame >= len(steps):
                continue
            capture_pose = poses[capture_frame, a]
            R = _rotation_from_wxyz(capture_pose).as_matrix()
            r_cpu = R.T @ (captures[a] - capture_pose[:3])
            capture_reconstruction_discrepancy.append(np.linalg.norm(r_cpu - offsets[a]))
            capture_frame_discrepancy.append(np.linalg.norm(actual[capture_frame, a] - captures[a]))
            for f in range(capture_frame, len(steps)):
                pose = poses[f, a]
                target[f, a] = pose[:3] + _rotation_from_wxyz(pose).as_matrix() @ r_cpu
                active[f, a] = True
        error = np.linalg.norm(actual - target, axis=-1)
        snapshots = self.snapshots
        saved = {"snapshot_tick": snapshots.tick.numpy(),
                 "snapshot_active": snapshots.active.numpy().reshape(-1, 4)}
        for name, width, shape in (("A", 16, (4, 4)), ("B", 16, (4, 4)),
                                   ("rhs", 12, (4, 3)), ("velocity", 12, (4, 3)),
                                   ("centroid", 12, (4, 3)), ("target", 12, (4, 3)),
                                   ("multiplier", 12, (4, 3)), ("pre", 12, (4, 3)),
                                   ("post", 12, (4, 3)), ("residual", 12, (4, 3))):
            saved[f"snapshot_{name}"] = getattr(snapshots, name).numpy().reshape(
                (self.snapshot_count,) + shape)
        saved["snapshot_centroid_world_m"] = to_world(saved["snapshot_centroid"])
        saved["snapshot_target_world_m"] = to_world(saved["snapshot_target"])
        for name in ("rhs", "velocity", "pre", "post", "residual"):
            saved[f"snapshot_{name}_world_mps"] = saved[f"snapshot_{name}"] * WORLD_TO_SIM
        conditions = {"raw": [], "equilibrated": []}
        diagnostic_failures = []
        for slot, tick in enumerate(saved["snapshot_tick"]):
            if tick < 0:
                continue
            active_rows = np.flatnonzero(saved["snapshot_active"][slot])
            for name, field in (("raw", "A"), ("equilibrated", "B")):
                matrix = saved[f"snapshot_{field}"][slot][np.ix_(active_rows, active_rows)]
                if not np.isfinite(matrix).all():
                    value = None
                    diagnostic_failures.append({"slot": slot, "tick": int(tick),
                                                "matrix": name, "status": "nonfinite"})
                else:
                    try:
                        number = float(np.linalg.cond(matrix))
                        value = number if np.isfinite(number) else None
                        if value is None:
                            diagnostic_failures.append({"slot": slot, "tick": int(tick),
                                                        "matrix": name, "status": "singular"})
                    except np.linalg.LinAlgError as exc:
                        value = None
                        diagnostic_failures.append({"slot": slot, "tick": int(tick),
                                                    "matrix": name, "status": str(exc)})
                conditions[name].append((int(tick), value))
        failure_code = int(self.work.failure.numpy()[0])
        max_residual = float(self.work.max_residual.numpy()[0])
        status = "numerical_failure" if failure_code else "ok"
        def worst(values):
            finite = [(tick, value) for tick, value in values if value is not None]
            if not finite:
                return None
            tick, value = max(finite, key=lambda item: item[1])
            return {"tick": tick,
                    "source_step": int(p["source_steps"][0] + tick // p["n_substeps"]),
                    "condition_number": value}
        summary = {
            "status": status, "failure_code": failure_code,
            "numerical_failure_count": int(failure_code != 0),
            "failed_tick": int(self.work.failed_tick.numpy()[0]),
            "max_active_grid_centroid_velocity_residual_world_mps": max_residual,
            "max_residual_tick": int(self.work.max_tick.numpy()[0]),
            "max_residual_patch": int(self.work.max_row.numpy()[0]),
            "max_independent_active_endpoint_error_mm": (
                float(np.nanmax(error[active]) * 1000) if active.any() else None),
            "max_capture_vs_returned_frame_discrepancy_mm": (
                float(max(capture_frame_discrepancy) * 1000) if capture_frame_discrepancy else None),
            "max_capture_reconstruction_discrepancy_mm": (
                float(max(capture_reconstruction_discrepancy) * 1000)
                if capture_reconstruction_discrepancy else None),
            "max_capture_target_discrepancy_mm": (
                float(np.max(np.linalg.norm(captures[captured] - capture_targets[captured], axis=1)) * 1000)
                if captured.any() else None),
            "worst_saved_raw_condition": worst(conditions["raw"]),
            "worst_saved_equilibrated_condition": worst(conditions["equilibrated"]),
            "condition_diagnostic_failures": diagnostic_failures,
            "completed_source_range": [int(steps[0]), int(steps[-1])],
        }
        return {
            "selected_actual_masses_sim": self.selected_masses,
            "patch_masses_sim": self.patch_masses,
            "captured": captured, "capture_world_m": captures,
            "capture_target_world_m": capture_targets,
            "capture_source_step": p["activation_steps"],
            "finger_local_centroid_offset_world_m": offsets,
            "actual_mass_centroid_world_m": actual,
            "target_mass_centroid_world_m": target,
            "active_endpoint_mask": active,
            "endpoint_centroid_error_mm": error * 1000,
            **saved, "summary": summary,
        }
