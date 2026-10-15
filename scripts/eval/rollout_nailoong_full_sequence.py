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
from scripts.eval.nailoong_recorded_finger_patches import (
    RecordedFingerMaterialPatches, prepare_recorded_finger_patches,
)
from scripts.eval.nailoong_recorded_finger_grid_centroids import RecordedFingerGridCentroids
from scripts.eval.nailoong_material_face_observations import (
    prepare_material_faces, reconstruct_material_faces,
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
    parser.add_argument("--arms", nargs="+", choices=("default", "spatial", "homogeneous"),
                        default=("default", "spatial"))
    parser.add_argument("--start-source-step", type=int, default=0,
                        help="Source state used to initialize the open-loop rollout")
    parser.add_argument("--end-source-step", type=int, default=None,
                        help="Inclusive final source state (default: last available)")
    parser.add_argument("--representation", choices=("legacy_surface", "matched_surface",
                                                   "tet_volume"), default="legacy_surface")
    parser.add_argument("--floor-surface", choices=("sticky", "frictional"), default=None,
                        help="Override the recorded floor surface for corrected representations")
    parser.add_argument("--floor-friction", type=float, default=None,
                        help="Override the recorded floor friction for corrected representations")
    parser.add_argument("--source-run", type=Path, default=Path(
        "/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6"))
    parser.add_argument("--controller-mode", choices=("grid_cubes", "recorded_finger_patches",
                                                      "recorded_finger_grid_centroids"),
                        default=None)
    parser.add_argument("--observation-readout", choices=("grid_markers", "material_faces"),
                        default=None)
    parser.add_argument("--observation-field", type=Path, default=None)
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
                  start_source_step, geometry, observation_labels,
                  recorded_controller_inputs=None, material_face_inputs=None):
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
            [(0., 0., 0.5 - trainer.mpm_shift[2]), (0., 0., 1.),
             args.floor_surface, args.floor_friction])
        if args.controller_mode == "grid_cubes":
            for i in range(trainer.n_ctrl_parts):
                point = np.asarray(trainer.ctrl_param["handler_pos"][i]) - trainer.mpm_shift
                mpm.moving_boundaries.append(
                    [point, trainer.ctrl_param["csize"][i], (0., 0., 0.), 0.0])
        mpm.initialize()
        mpm.mpm_solver.profile_substeps = getattr(args, "profile_substeps", True)
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
    readout_mode = getattr(args, "observation_readout", "grid_markers")
    owner_F_frames = None
    owner_initial_world_m = None
    if readout_mode == "material_faces":
        owner_full_ids = material_face_inputs["material_face_unique_owner_full_ids"]
        owner_initial_world_m = initial_world[owner_full_ids].copy()
        owner_indices = torch.as_tensor(owner_full_ids, dtype=torch.long, device=mpm.x_t.device)
        owner_F_frames = []

        def observe_frame(state, frame_index):
            owner_F = wp.to_torch(state.particle_F_trial)[owner_indices]
            owner_F_frames.append(owner_F.to(dtype=torch.float32).cpu().numpy().copy())

        mpm.frame_state_observer = observe_frame
    if args.controller_mode == "grid_cubes":
        attach_timed_controllers(mpm, trainer, start_source_step)
    elif args.controller_mode == "recorded_finger_patches":
        mpm.particle_boundary = RecordedFingerMaterialPatches(recorded_controller_inputs)
    elif args.controller_mode == "recorded_finger_grid_centroids":
        mpm.grid_boundary = RecordedFingerGridCentroids(
            recorded_controller_inputs, mpm.mpm_state, mpm.mpm_model,
            mpm.mpm_solver.collider_params[0].point)
    sim_positions = mpm.advance_to(frame_count - 1)
    shifted = sim_positions + torch.as_tensor(trainer.mpm_shift,
                                              dtype=sim_positions.dtype,
                                              device=sim_positions.device)
    full_prediction = inv_transform_points(shifted, name=trainer.exp_name).cpu().numpy()
    readout_summary = {"mode": readout_mode, "status": "ok",
                       "completed_source_range": [start_source_step,
                                                  start_source_step + len(full_prediction) - 1]}
    if readout_mode == "material_faces":
        owner_F_trial = np.stack(owner_F_frames)
        owner_world_m = full_prediction[:, owner_full_ids]
        selected_prediction = reconstruct_material_faces(
            material_face_inputs, owner_initial_world_m, owner_world_m, owner_F_trial)
        if isinstance(selected_prediction, dict):
            failure = selected_prediction
            selected_prediction = None
            readout_summary.update(failure)
            readout_summary["status"] = "numerical_failure"
            readout_summary["first_failed_source_frame"] = (
                start_source_step + failure["first_failed_frame"])
            print(f"WARNING: material-face observation readout failed at source frame "
                  f"{readout_summary['first_failed_source_frame']}: "
                  f"{failure['component']}: {failure['reason']}",
                  flush=True)
        initialized["observation_readout_result"] = {
            "prediction_world_m": selected_prediction,
            "owner_F_trial": owner_F_trial,
            "owner_initial_world_m": owner_initial_world_m,
            "summary": readout_summary,
        }
    else:
        initialized["observation_readout_result"] = {
            "prediction_world_m": full_prediction[:, :len(observation_labels)],
            "summary": readout_summary,
        }
    if args.controller_mode == "recorded_finger_patches":
        initialized["controller_diagnostics"] = mpm.particle_boundary.diagnostics(full_prediction)
    elif args.controller_mode == "recorded_finger_grid_centroids":
        diagnostics = mpm.grid_boundary.diagnostics(full_prediction)
        diagnostics["summary"].update({
            "grid_collider_count": len(mpm.mpm_solver.collider_params),
            "grid_cuboid_count": len(mpm.moving_boundaries),
            "particle_velocity_modifier_count": len(mpm.mpm_solver.particle_velocity_modifiers),
            "pre_p2g_operation_count": len(mpm.mpm_solver.pre_p2g_operations),
            "particle_boundary_present": mpm.particle_boundary is not None,
        })
        initialized["controller_diagnostics"] = diagnostics
    return full_prediction, initialized


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
    args.observation_readout = (cli.observation_readout or
                                fit_summary["config"].get("observation_readout", "grid_markers"))
    args.observation_field = (cli.observation_field or Path(
        fit_summary["config"].get("observation_field",
                                  "outputs/nailoong_material_init/field.npz")))
    if (args.observation_readout == "material_faces" and
            (args.representation != "tet_volume" or cli.start_source_step != 0)):
        print("Unsupported material_faces readout: requires tet_volume initialized at "
              "source state 0", flush=True)
        return
    args.controller_mode = (cli.controller_mode or
                            fit_summary["config"].get("controller_mode", "grid_cubes"))
    if args.controller_mode == "grid_cubes":
        # Recorded-patch fits omit cube settings; an explicit cube override uses H1 defaults.
        args.controller_halfsize = fit_summary["config"].get("controller_halfsize", .003)
        args.controller_anchor = fit_summary["config"].get("controller_anchor", "semantic")
        args.controller_vscale = fit_summary["config"].get("controller_vscale", None)
    if args.controller_mode in ("recorded_finger_patches", "recorded_finger_grid_centroids") and (
            args.representation != "tet_volume" or cli.start_source_step > 39):
        raise ValueError("recorded finger modes require tet_volume and a start at or before source state 39")
    args.floor_surface = (cli.floor_surface if cli.floor_surface is not None else
                          fit_summary["config"].get("floor_surface", "sticky"))
    args.floor_friction = (cli.floor_friction if cli.floor_friction is not None else
                           fit_summary["config"].get("floor_friction", 0.0))
    if args.representation == "legacy_surface" and (args.floor_surface != "sticky" or
                                                     args.floor_friction != 0.0):
        print("WARNING: corrected floor settings do not apply to legacy_surface; "
              "using sticky/0", flush=True)
        args.floor_surface = "sticky"
        args.floor_friction = 0.0
    with np.load(cli.source_tracks) as archive:
        source = {key: archive[key] for key in archive.files}
    start_index = int(np.flatnonzero(source["source_steps"] == cli.start_source_step)[0])
    end_index = (len(source["source_steps"]) - 1 if cli.end_source_step is None else
                 int(np.flatnonzero(source["source_steps"] == cli.end_source_step)[0]))
    for key in ("object_points_world_m", "object_velocities_world_mps",
                "controller_points_world_m", "object_visibilities", "source_steps",
                "time_s", "rigid_poses"):
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
    material_face_inputs = (prepare_material_faces(
        cli.source_run, args.observation_field, target[0], labels)
        if args.observation_readout == "material_faces" else None)

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
    recorded_controller_inputs = None
    if args.controller_mode == "grid_cubes":
        configure_controllers(trainer, labels, args)
    else:
        recorded_controller_inputs = prepare_recorded_finger_patches(
            cli.source_run, source["source_steps"], source["rigid_poses"],
            len(labels), args.sim_shift, args.frame_dt, fit_summary["actual_sim_dt"],
            cfg.device)
    moduli = np.load(cli.run_dir / "moduli.npz")
    trajectories = {"gt_world_m": target,
                    "controller_world_m": source["controller_points_world_m"],
                    "part_labels": labels,
                    "source_steps": source["source_steps"],
                    "time_s": source["time_s"],
                    "object_visibilities": visible}
    if material_face_inputs is not None:
        trajectories.update(material_face_inputs)
    if recorded_controller_inputs is not None:
        prepared = recorded_controller_inputs
        trajectories.update({
            "controller_patch_ids": prepared["patch_ids"],
            "controller_tet_ids": prepared["tet_ids"],
            "controller_full_ids": prepared["full_ids"],
            "controller_patch_links": prepared["links"],
            "controller_patch_part_labels": prepared["part_labels"],
            "controller_patch_activation_source_steps": prepared["activation_steps"],
            "controller_rigid_poses_xyz_wxyz": prepared["poses_xyz_wxyz"],
        })
    rows = []
    per_frame = []
    result = {
        "source_tracks": str(cli.source_tracks), "source_fit": str(cli.run_dir),
        "source_run": str(cli.source_run),
        "source_initial_state": str(cli.source_run / "initial_state.npz"),
        "source_selected_state": str(cli.source_run / "states" /
                                     f"{cli.start_source_step:06d}.npz"),
        "representation": cli.representation,
        "observation_readout": args.observation_readout,
        "observation_field": str(args.observation_field),
        "observation_prediction": ("material-attached source0 face reconstruction from "
            "actual initialized owner centers and completed endpoint F_trial, followed "
            "by proper triangle Kabsch transfer" if material_face_inputs is not None else
            "independently advected grid markers"),
        "legacy_grid_markers": "retained as unchanged independently advected diagnostics",
        "observation_correspondence": ("fixed privileged synthetic source face associations "
            "shared by both fit arms; no future object targets used for readout"),
        "inference": "forward numerical inference; full source0--149 includes training frames",
        "holdout_disclosure": ("source135--149 did not select fitted parameters but was "
                               "previously inspected during model debugging"),
        "material_prior_disclosure": "source material asset supplied the semantic prior",
        "particle_count": len(labels), "frame_count": frame_count,
        "source_steps": [int(source["source_steps"][0]), int(source["source_steps"][-1])],
        "initial_time_s": float(source["time_s"][0]),
        "frame_dt_s": args.frame_dt, "mpm_substeps_per_frame":
            round(args.frame_dt / fit_summary["actual_sim_dt"]),
        "mpm_substep_dt_s": fit_summary["actual_sim_dt"],
        "grid_velocity_damping_per_substep": args.grid_damping,
        "grid_size": args.grid_size, "grid_lim_sim_units": args.grid_lim,
        "sim_shift": args.sim_shift, "controller_mode": args.controller_mode,
        "grid_cuboid_count": 2 if args.controller_mode == "grid_cubes" else 0,
        "grid_collider_count": 3 if args.controller_mode == "grid_cubes" else 1,
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
        "floor": (f"{args.floor_surface} plane at world z=0, "
                  f"friction {args.floor_friction:g}"),
        "floor_surface": args.floor_surface,
        "floor_friction": args.floor_friction,
        "floor_sim_z": 0.5 - trainer.mpm_shift[2],
        "material_fields": "unchanged fitted checkpoint fields; no optimization",
        "results": {},
    }
    if recorded_controller_inputs is None:
        result.update({
            "controller_halfsize_sim_units": args.controller_halfsize,
            "controller_activation": {
                "belly": "transition source state 39 to 40",
                "ring": "transition source state 40 to 41"},
            "controller_anchors": ("nearest same-part particle at initial source state; "
                                   "follows gripper centers"),
            "controller_initial_anchor_sim_unshifted": [np.asarray(p).tolist()
                                                          for p in trainer.ctrl_param["handler_pos"]],
            "controller_velocity_scales": trainer.ctrl_param["vscale"],
        })
    else:
        result.update({
            "source_snap_bindings": str(cli.source_run / "snap_bindings.json"),
            "controller_pose_source": f"{cli.source_tracks}:rigid_poses",
            "controller_pose_convention": "world xyz+wxyz; wxyz reordered to xyzw for SciPy",
            "controller_interpolation": "linear translation and shortest-arc SciPy Slerp at integer substep ticks",
            "controller_patch_definitions": recorded_controller_inputs["patches"],
            "controller_offset_units": "world metres; simulated offsets divide by 5",
            "controller_approximation": ("kinematic hard attachment to four source finger links; "
                                         "source snaps had finite stiffness and native contact"),
            "controller_contact_association": ("privileged synthetic source binding metadata "
                                               "for 13 vertex/link associations, shared by both "
                                               "potential fit arms; no future object targets"),
        })
        if args.controller_mode == "recorded_finger_grid_centroids":
            result["controller_approximation"] = (
                "mass-weighted patch-centroid grid projection, actual capture from simulated "
                "states 39/40; hard centroids approximate finite source snaps and native contact")
            result["controller_grid_projection"] = (
                "joint exact float64 solve with diagonal equilibration after frictional floor; "
                "correction only above floor at grid mass >1e-15; no particle writes; "
                "ordinary marker advection; forward-only optimization")
    for evaluation in cli.evaluations:
        for arm in cli.arms:
            key = f"{arm}_eval{evaluation}"
            field_key = f"{key}_pa"
            if evaluation == 0 and field_key not in moduli.files:
                field_key = f"{arm}_initial_pa"
            E_pa = moduli[field_key]
            started = time.perf_counter()
            set_all_seeds(args.seed)
            full_prediction, initialized = fixed_rollout(
                trainer, E_pa, velocity, args, frame_count, cli.start_source_step,
                geometry, labels, recorded_controller_inputs, material_face_inputs)
            diagnostics = initialized.pop("controller_diagnostics", None)
            readout = initialized.pop("observation_readout_result")
            if diagnostics is not None:
                for diagnostic_name, values in diagnostics.items():
                    if diagnostic_name != "summary":
                        trajectories[f"{key}_controller_{diagnostic_name}"] = values
            prediction = readout["prediction_world_m"]
            material_mask = initialized["initial_full_particle_volumes_sim"] > 0
            material_labels = initialized["initial_full_particle_part_labels"][material_mask]
            if prediction is not None:
                trajectories[f"{key}_world_m"] = prediction
            trajectories[f"{key}_grid_marker_world_m"] = full_prediction[:, :len(labels)]
            trajectories[f"{key}_material_world_m"] = full_prediction[:, material_mask]
            trajectories[f"{key}_observation_readout_summary_json"] = np.array(
                json.dumps(readout["summary"]))
            if material_face_inputs is not None:
                trajectories[f"{key}_owner_F_trial"] = readout["owner_F_trial"]
                trajectories[f"{key}_owner_initial_world_m"] = readout["owner_initial_world_m"]
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
            controller_failed = (diagnostics is not None and
                                 diagnostics["summary"].get("status") == "numerical_failure")
            readout_failed = readout["summary"]["status"] == "numerical_failure"
            if controller_failed or readout_failed:
                raw_range = [cli.start_source_step,
                             cli.start_source_step + len(full_prediction) - 1]
                result["results"][key] = {
                    "status": "numerical_failure",
                    "numerical_failure_component": ("grid_projection" if controller_failed
                                                    else "observation_readout"),
                    "raw_completed_source_range": raw_range,
                    "completed_source_range": raw_range,
                    "observation_readout": readout["summary"],
                    "wall_time_s": time.perf_counter() - started,
                }
                if diagnostics is not None:
                    result["results"][key]["controller_diagnostics"] = diagnostics["summary"]
                result["status"] = "numerical_failure"
                result["numerical_failure_component"] = result["results"][key][
                    "numerical_failure_component"]
                result["raw_completed_source_range"] = raw_range
                result["completed_source_range"] = raw_range
                np.savez_compressed(cli.output_dir / "trajectories.npz", **trajectories)
                (cli.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
                print(f"{key}: {result['numerical_failure_component']} numerical failure; "
                      f"raw prefix ends at source state {raw_range[1]}",
                      flush=True)
                return
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
                "observation_readout": readout["summary"],
            }
            if diagnostics is not None:
                result["results"][key]["controller_diagnostics"] = diagnostics["summary"]
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
