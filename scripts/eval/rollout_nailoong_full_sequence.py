"""Replay fixed Nailoong fits over the source sequence with timed grasps.

The source-track archive maps the original 1765 observed particle identities to
every FEM state. Only the selected initial state initializes MPM; later object
states are targets for evaluation. Controllers follow the source gripper tracks.
"""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import warp as wp
from scipy.spatial import cKDTree

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.diff_simulator import WarpMPMWrapper
from phys_sim.engine.utils import inv_transform_points, transform_points
from phys_sim.utils import cfg
from scripts.train_test.run_nailoong_material_init import (
    PART_NAMES, configure_controllers, make_mpm,
)
from scripts.utils import set_all_seeds


PHASES = {
    "open": (0, 30),
    "grasp": (30, 50),
    "lift": (50, 80),
    "squeeze": (80, 115),
    "pull": (115, 150),
}
# Controller index 0 is the ring gripper; index 1 is the belly gripper.
# Source state k was saved after control step k. The belly/ring attachments
# occur before source steps 40/41, respectively.
FIRST_ACTIVE_TRANSITION = {0: 40, 1: 39}
WORLD_TO_SIM_LENGTH_DIVISOR = 5.0
WORLD_GRAVITY_M_PER_S2 = (0.0, 0.0, -9.81)
SIM_GRAVITY_UNITS_PER_S2 = tuple(
    component / WORLD_TO_SIM_LENGTH_DIVISOR for component in WORLD_GRAVITY_M_PER_S2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tracks", type=Path,
                        default=Path("outputs/nailoong_full_sequence/source_tracks.npz"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/nailoong_material_init/experiment_refined"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/nailoong_full_sequence/rollout"))
    parser.add_argument("--evaluations", type=int, nargs="+", default=(8,))
    parser.add_argument("--arms", nargs="+", choices=("default", "spatial"),
                        default=("default", "spatial"))
    parser.add_argument("--start-source-step", type=int, default=0,
                        help="Source state used to initialize the open-loop rollout")
    parser.add_argument("--end-source-step", type=int, default=None,
                        help="Inclusive final source state (default: last available)")
    parser.add_argument("--representation", choices=("legacy_surface", "matched_surface",
                                                   "tet_volume"), default="legacy_surface")
    parser.add_argument("--source-run", type=Path, default=Path(
        "/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6"))
    return parser.parse_args()


def extended_case(source, original_case, output_dir):
    with original_case.open("rb") as stream:
        data = pickle.load(stream)
    object_points = source["object_points_world_m"].astype(np.float32)
    controller_points = source["controller_points_world_m"].astype(np.float32)
    frame_count, particle_count, _ = object_points.shape
    assert particle_count == data["object_points"].shape[1]
    assert controller_points.shape == (frame_count, data["controller_points"].shape[1], 3)
    assert np.array_equal(np.diff(source["source_steps"]), np.ones(frame_count - 1))
    data["object_points"] = object_points
    data["controller_points"] = controller_points
    data["object_colors"] = np.repeat(data["object_colors"][:1], frame_count, axis=0)
    data["object_visibilities"] = source["object_visibilities"].astype(bool)
    data["object_motions_valid"] = source["object_visibilities"].astype(bool)
    case_path = output_dir / "full_sequence_data.pkl"
    with case_path.open("wb") as stream:
        pickle.dump(data, stream)
    return case_path


def attach_timed_controllers(mpm, trainer, start_source_step):
    anchors = [np.asarray(point) for point in trainer.ctrl_param["handler_pos"]]
    masks = trainer.ctrl_param["ctrl_mask"]
    centers = torch.stack([trainer.ctrl_points[:, masks == i].mean(dim=1)
                           for i in range(trainer.n_ctrl_parts)], dim=1)
    center_origin = centers[0].clone()
    shift = np.asarray(trainer.mpm_shift)

    def callback(wrapper, frame):
        source_transition = start_source_step + frame
        for i in range(trainer.n_ctrl_parts):
            collider = wrapper.mpm_solver.collider_params[i + 1]
            position = anchors[i] + (centers[frame, i] - center_origin[i]).cpu().numpy() - shift
            velocity = ((centers[frame + 1, i] - centers[frame, i])
                        * trainer.ctrl_param["vscale"][i][0]).cpu().numpy()
            collider.point = wp.vec3(*position)
            collider.velocity = wp.vec3(*velocity)
            collider.start_time = (0.0 if source_transition >= FIRST_ACTIVE_TRANSITION[i]
                                   else 999.0)

    mpm.frame_callback = callback


def tet_geometry(source_run, start_source_step, observations_world, observation_labels):
    with np.load(source_run / "initial_state.npz") as initial:
        tets = initial["tetrahedra"]
        tet_labels = initial["tet_part_labels"].astype(np.int64)
    with np.load(source_run / "states" / f"{start_source_step:06d}.npz") as state:
        vertices = state["positions_m"].astype(np.float64)
        velocities = state["velocities_mps"].astype(np.float64)
    corners = vertices[tets]
    centroid_world = corners.mean(axis=1)
    velocity_world = velocities[tets].mean(axis=1)
    edges = corners[:, 1:] - corners[:, :1]
    volumes_sim = np.abs(np.linalg.det(edges)) / (6.0 * WORLD_TO_SIM_LENGTH_DIVISOR ** 3)

    matched_volumes = np.zeros(len(observations_world), dtype=np.float64)
    for label in PART_NAMES:
        observed = np.flatnonzero(observation_labels == label)
        tet_indices = np.flatnonzero(tet_labels == label)
        nearest = cKDTree(observations_world[observed].astype(np.float64)).query(
            centroid_world[tet_indices])[1]
        np.add.at(matched_volumes, observed[nearest], volumes_sim[tet_indices])
    return centroid_world, velocity_world, tet_labels, volumes_sim, matched_volumes


def fixed_rollout(trainer, E_pa, initial_velocity_world, args, frame_count,
                  start_source_step, geometry, observation_labels):
    # Preserve the exact mapped source observations and instantaneous velocity.
    initial_x = transform_points(trainer.object_points[0].clone(), name=trainer.exp_name)
    initial_x -= torch.as_tensor(trainer.mpm_shift, dtype=initial_x.dtype,
                                 device=initial_x.device)
    initial_v = torch.as_tensor(initial_velocity_world / WORLD_TO_SIM_LENGTH_DIVISOR,
                                dtype=initial_x.dtype,
                                device=initial_x.device)
    if args.representation == "legacy_surface":
        mpm = make_mpm(trainer, E_pa, args, diff_sim=False)
        # make_mpm/initialize_mpm otherwise lift the cloud by 5 mm.
        mpm.x_t = initial_x
        mpm.v_t = initial_v
        mpm.mpm_state.reset_state(initial_x, None, initial_v, device=mpm.device,
                                  requires_grad=False)
        particle_labels = observation_labels
    else:
        centroid_world, velocity_world, tet_labels, tet_volumes, matched_volumes = geometry
        if args.representation == "tet_volume":
            centroid_x = transform_points(
                torch.as_tensor(centroid_world, dtype=initial_x.dtype, device=initial_x.device),
                name=trainer.exp_name)
            centroid_x -= torch.as_tensor(trainer.mpm_shift, dtype=initial_x.dtype,
                                          device=initial_x.device)
            centroid_v = torch.as_tensor(velocity_world / WORLD_TO_SIM_LENGTH_DIVISOR,
                                         dtype=initial_v.dtype,
                                         device=initial_v.device)
            initial_x = torch.cat((initial_x, centroid_x))
            initial_v = torch.cat((initial_v, centroid_v))
            volumes = np.concatenate((np.zeros(len(observation_labels)), tet_volumes))
            particle_labels = np.concatenate((observation_labels, tet_labels))
            E_full = np.concatenate((E_pa, np.where(tet_labels == 0,
                E_pa[observation_labels == 0][0],
                E_pa[observation_labels != 0][0])))
        else:
            volumes = matched_volumes
            particle_labels = observation_labels
            E_full = E_pa

        mpm = WarpMPMWrapper(requires_grad=False, **trainer.mpm_kwargs)
        mpm.material_params = {"material": "jelly"}
        mpm.add_particles(initial_x, initial_v, volumes, E_full, 0.3, 2000.0)
        mpm.surface_colliders.append(
            [(0., 0., 0.5 - trainer.mpm_shift[2]), (0., 0., 1.), "sticky", 0.])
        for i in range(trainer.n_ctrl_parts):
            point = np.asarray(trainer.ctrl_param["handler_pos"][i]) - trainer.mpm_shift
            mpm.moving_boundaries.append(
                [point, trainer.ctrl_param["csize"][i], (0., 0., 0.), 0.0])
        mpm.initialize()
        mpm.mpm_model.grid_v_damping_scale = args.grid_damping
        mpm.mpm_solver.time = 0.0

    mpm.gravity = list(SIM_GRAVITY_UNITS_PER_S2)
    mpm.mpm_model.gravitational_accelaration = wp.vec3(*SIM_GRAVITY_UNITS_PER_S2)

    initial_shifted = mpm.x_t + torch.as_tensor(trainer.mpm_shift,
                                                 dtype=mpm.x_t.dtype, device=mpm.x_t.device)
    initial_world = inv_transform_points(initial_shifted, name=trainer.exp_name).cpu().numpy()
    initial_velocity_world = (mpm.v_t * WORLD_TO_SIM_LENGTH_DIVISOR).cpu().numpy()
    initialized = {
        "initial_full_particle_world_m": initial_world,
        "initial_full_particle_velocity_world_mps": initial_velocity_world,
        "initial_full_particle_volumes_sim": mpm.vol_t.cpu().numpy(),
        "initial_full_particle_part_labels": particle_labels,
        "initial_full_particle_E_pa": mpm.E_t.cpu().numpy(),
    }
    attach_timed_controllers(mpm, trainer, start_source_step)
    sim_positions = mpm.advance_to(frame_count - 1)
    shifted = sim_positions + torch.as_tensor(trainer.mpm_shift,
                                              dtype=sim_positions.dtype,
                                              device=sim_positions.device)
    return inv_transform_points(shifted, name=trainer.exp_name).cpu().numpy(), initialized


def body_height_mm(points, body_mask):
    z = points[body_mask, 2] * 1000.0
    z05, z50, z95 = np.percentile(z, (5, 50, 95))
    return {"z05_mm": float(z05), "z50_mm": float(z50),
            "z95_mm": float(z95), "z95_minus_z05_mm": float(z95 - z05)}


def metrics_for(name, prediction, target, visible, labels, source_steps):
    distance_mm = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    rows = []
    for phase, (start, stop) in {"all": (int(source_steps[0]),
                                           int(source_steps[-1]) + 1), **PHASES}.items():
        frame_mask = (source_steps >= start) & (source_steps < stop)
        if not frame_mask.any():
            continue
        for part, part_name in [(None, "all"), *PART_NAMES.items()]:
            mask = visible[frame_mask].copy()
            if part is not None:
                mask &= labels[None, :] == part
            values = distance_mm[frame_mask][mask]
            rows.append({"arm": name, "phase": phase, "part": part_name,
                         "count": int(values.size),
                         "rmse_mm": float(np.sqrt(np.mean(values ** 2))),
                         "mean_error_mm": float(np.mean(values)),
                         "max_error_mm": float(np.max(values))})
    return rows, distance_mm


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with (cli.run_dir / "summary.json").open() as stream:
        fit_summary = json.load(stream)
    args = SimpleNamespace(**fit_summary["config"])
    args.start_frame = 0
    args.case_dir = Path(args.case_dir)
    args.grid_damping = fit_summary["grid_velocity_damping"]
    args.representation = cli.representation
    with np.load(cli.source_tracks) as archive:
        source = {key: archive[key] for key in archive.files}
    start_index = int(np.flatnonzero(source["source_steps"] == cli.start_source_step)[0])
    end_index = (len(source["source_steps"]) - 1 if cli.end_source_step is None else
                 int(np.flatnonzero(source["source_steps"] == cli.end_source_step)[0]))
    for key in ("object_points_world_m", "object_velocities_world_mps",
                "controller_points_world_m", "object_visibilities", "source_steps",
                "time_s"):
        source[key] = source[key][start_index:end_index + 1]
    frame_count = len(source["source_steps"])
    assert np.isclose(float(source["fps"]), 1.0 / args.frame_dt)
    case_path = extended_case(source, args.case_dir / "final_data.pkl", cli.output_dir)
    labels = source["part_labels"].astype(np.int64)
    target = source["object_points_world_m"].astype(np.float32)
    visible = source["object_visibilities"].astype(bool)
    velocity = source["object_velocities_world_mps"].astype(np.float32)[0]
    geometry = (tet_geometry(cli.source_run, cli.start_source_step, target[0], labels)
                if cli.representation != "legacy_surface" else None)

    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = args.case_dir.name
    cfg.train_frame = frame_count
    set_all_seeds(args.seed)
    trainer = InvPhyTrainerWarpMPM(
        data_path=str(case_path), base_dir=str(cli.output_dir),
        train_frame=frame_count, save_artifacts=False,
        mpm_kwargs={"grid_size": args.grid_size, "grid_lim": args.grid_lim,
                    "frame_dt": args.frame_dt,
                    "sim_dt": fit_summary["actual_sim_dt"]},
        mpm_shift=args.sim_shift,
    )
    configure_controllers(trainer, labels, args)
    moduli = np.load(cli.run_dir / "moduli.npz")
    trajectories = {"gt_world_m": target,
                    "controller_world_m": source["controller_points_world_m"],
                    "part_labels": labels,
                    "source_steps": source["source_steps"],
                    "time_s": source["time_s"],
                    "object_visibilities": visible}
    rows = []
    per_frame = []
    result = {
        "source_tracks": str(cli.source_tracks), "source_fit": str(cli.run_dir),
        "source_run": str(cli.source_run),
        "source_initial_state": str(cli.source_run / "initial_state.npz"),
        "source_selected_state": str(cli.source_run / "states" /
                                     f"{cli.start_source_step:06d}.npz"),
        "representation": cli.representation,
        "particle_count": len(labels), "frame_count": frame_count,
        "source_steps": [int(source["source_steps"][0]), int(source["source_steps"][-1])],
        "initial_time_s": float(source["time_s"][0]),
        "frame_dt_s": args.frame_dt, "mpm_substeps_per_frame":
            round(args.frame_dt / fit_summary["actual_sim_dt"]),
        "mpm_substep_dt_s": fit_summary["actual_sim_dt"],
        "grid_velocity_damping_per_substep": args.grid_damping,
        "grid_size": args.grid_size, "grid_lim_sim_units": args.grid_lim,
        "sim_shift": args.sim_shift, "controller_halfsize_sim_units":
            args.controller_halfsize,
        "density_parameter": 2000.0, "poisson_ratio": 0.3,
        "material_law": "jelly", "initial_deformation_gradient": "identity",
        "initial_affine_velocity": "zero",
        "gravity_sim_units_per_s2": list(SIM_GRAVITY_UNITS_PER_S2),
        "gravity_world_m_per_s2": list(WORLD_GRAVITY_M_PER_S2),
        "world_to_sim_length_divisor": WORLD_TO_SIM_LENGTH_DIVISOR,
        "volume_units": ("simulator length cubed (world metres divided by "
                         f"{WORLD_TO_SIM_LENGTH_DIVISOR:g} per axis)"),
        "mass_units": "density parameter times simulator volume; not physical kg",
        "initial_state": (f"mapped FEM source state {cli.start_source_step} positions "
                          "and instantaneous velocities, no floor lift"),
        "floor": "existing sticky plane at world z=0",
        "floor_sim_z": 0.5 - trainer.mpm_shift[2],
        "controller_activation": {
            "belly": "transition source state 39 to 40",
            "ring": "transition source state 40 to 41"},
        "controller_anchors": ("nearest same-part particle at initial source state; "
                               "follows gripper centers"),
        "controller_initial_anchor_sim_unshifted": [np.asarray(p).tolist()
                                                      for p in trainer.ctrl_param["handler_pos"]],
        "controller_velocity_scales": trainer.ctrl_param["vscale"],
        "material_fields": "unchanged fitted checkpoint fields; no optimization",
        "results": {},
    }
    for evaluation in cli.evaluations:
        for arm in cli.arms:
            key = f"{arm}_eval{evaluation}"
            E_pa = moduli[f"{key}_pa"]
            started = time.perf_counter()
            set_all_seeds(args.seed)
            full_prediction, initialized = fixed_rollout(
                trainer, E_pa, velocity, args, frame_count, cli.start_source_step,
                geometry, labels)
            prediction = full_prediction[:, :len(labels)]
            material_mask = initialized["initial_full_particle_volumes_sim"] > 0
            material_labels = initialized["initial_full_particle_part_labels"][material_mask]
            trajectories[f"{key}_world_m"] = prediction
            trajectories[f"{key}_material_world_m"] = full_prediction[:, material_mask]
            trajectories[f"{key}_initial_full_particle_E_pa"] = initialized.pop(
                "initial_full_particle_E_pa")
            if "initial_full_particle_world_m" not in trajectories:
                trajectories.update(initialized)
                trajectories["material_part_labels"] = material_labels
                trajectories["material_volumes_sim"] = (
                    initialized["initial_full_particle_volumes_sim"][material_mask])
                volumes = initialized["initial_full_particle_volumes_sim"]
                particle_labels = initialized["initial_full_particle_part_labels"]
                result["total_particle_count"] = int(len(volumes))
                result["observation_count"] = int(len(labels))
                result["positive_volume_particle_count"] = int(material_mask.sum())
                result["material_total_volume_sim_length_cubed"] = float(volumes.sum())
                result["material_total_mass_rho_times_sim_volume"] = float(2000.0 * volumes.sum())
                result["material_per_part"] = {
                    name: {
                        "positive_volume_particle_count": int(np.count_nonzero(
                            material_mask & (particle_labels == label))),
                        "volume_sim_length_cubed": float(volumes[particle_labels == label].sum()),
                        "mass_rho_times_sim_volume": float(
                            2000.0 * volumes[particle_labels == label].sum()),
                    } for label, name in PART_NAMES.items()}
            arm_rows, distance_mm = metrics_for(key, prediction, target, visible,
                                                labels, source["source_steps"])
            rows.extend(arm_rows)
            for frame in range(frame_count):
                values = distance_mm[frame, visible[frame]]
                row = {"arm": key,
                       "source_step": int(source["source_steps"][frame]),
                       "time_s": float(source["time_s"][frame]),
                       "rmse_mm": float(np.sqrt(np.mean(values ** 2))),
                       "max_error_mm": float(np.max(values))}
                for prefix, points, body_mask in (
                        ("pred_body", prediction[frame], labels == 0),
                        ("target_body", target[frame], labels == 0),
                        ("material_body_unweighted", full_prediction[frame, material_mask],
                         material_labels == 0)):
                    row.update({f"{prefix}_{metric}": value for metric, value in
                                body_height_mm(points, body_mask).items()})
                per_frame.append(row)
            result["results"][key] = {
                "body_E_pa": float(E_pa[labels == 0][0]),
                "hardware_E_pa": float(E_pa[labels != 0][0]),
                "wall_time_s": time.perf_counter() - started,
                "overall_rmse_mm": arm_rows[0]["rmse_mm"],
                "final_frame_rmse_mm": per_frame[-1]["rmse_mm"],
            }
            np.savez_compressed(cli.output_dir / "trajectories.npz", **trajectories)
            (cli.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{key}: overall RMSE {arm_rows[0]['rmse_mm']:.3f} mm, "
                  f"final RMSE {per_frame[-1]['rmse_mm']:.3f} mm", flush=True)
    with (cli.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (cli.output_dir / "frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_frame[0].keys())
        writer.writeheader()
        writer.writerows(per_frame)


if __name__ == "__main__":
    main()
