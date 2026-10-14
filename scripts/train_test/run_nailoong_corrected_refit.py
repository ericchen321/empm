"""Fit paired two-material fields with the corrected continuous Nailoong rollout.

This is forward numerical optimization of EMPM, not native Adam/MyTape training.
Only source states 1--134 supply object targets to the training objective.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import minimize

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import cfg
from scripts.eval.rollout_nailoong_full_sequence import (
    extended_case, fixed_rollout, tet_geometry,
)
from scripts.eval.nailoong_recorded_finger_patches import prepare_recorded_finger_patches
from scripts.eval.nailoong_material_face_observations import prepare_material_faces
from scripts.train_test.run_nailoong_material_init import configure_controllers
from scripts.utils import set_all_seeds


TRAIN_FRAME_COUNT = 135
MAX_EVALUATIONS = 48
SIM_DT = 9.98003992015968e-5
HISTORY_FIELDS = (
    "arm", "call", "log_body_E_pa", "log_hardware_E_pa", "body_E_pa",
    "hardware_E_pa", "train_mse_mm2", "grid_marker_train_mse_mm2",
    "grid_marker_status", "readout_status", "best_train_mse_mm2", "best_call",
    "candidate_duration_s", "elapsed_arm_wall_s",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--source-tracks", type=Path,
                        default=Path("outputs/nailoong_full_sequence/source_tracks.npz"))
    parser.add_argument("--source-run", type=Path, default=Path(
        "/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-mode", choices=("grid_cubes", "recorded_finger_patches",
                                                      "recorded_finger_grid_centroids"),
                        default="grid_cubes")
    parser.add_argument("--observation-readout", choices=("grid_markers", "material_faces"),
                        default="grid_markers")
    parser.add_argument("--observation-field", type=Path,
                        default=Path("outputs/nailoong_material_init/field.npz"))
    return parser.parse_args()


def material_field(log_E, labels):
    E = np.exp(log_E)
    return np.where(labels == 0, E[0], E[1]).astype(np.float32)


def save_summary(path, summary):
    with path.open("w") as stream:
        json.dump(summary, stream, indent=2)


def main():
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cli.source_tracks) as archive:
        source = {key: archive[key] for key in archive.files}
    for key in ("object_points_world_m", "object_velocities_world_mps",
                "controller_points_world_m", "object_visibilities", "source_steps",
                "time_s", "rigid_poses"):
        source[key] = source[key][:TRAIN_FRAME_COUNT]

    case_path = extended_case(source, cli.case_dir / "final_data.pkl", cli.output_dir)
    labels = source["part_labels"].astype(np.int64)
    target = source["object_points_world_m"].astype(np.float32)
    visible = source["object_visibilities"].astype(bool)
    velocity = source["object_velocities_world_mps"][0].astype(np.float32)
    geometry = tet_geometry(cli.source_run, 0, target[0], labels)
    material_face_inputs = (prepare_material_faces(
        cli.source_run, cli.observation_field, target[0], labels)
        if cli.observation_readout == "material_faces" else None)
    tet_labels, tet_volumes = geometry[2], geometry[3]
    body_volume_fraction = float(tet_volumes[tet_labels == 0].sum() /
                                 tet_volumes.sum())
    semantic_body_E = 8000.0 / (body_volume_fraction +
                                440.0 * (1.0 - body_volume_fraction))
    seed_E = {
        "default": np.array([8000.0, 8000.0]),
        "spatial": np.array([semantic_body_E, 440.0 * semantic_body_E]),
    }

    args = SimpleNamespace(
        case_dir=cli.case_dir, grid_size=32, grid_lim=.064,
        sim_shift=(.54, .47, .496), frame_dt=1 / 30, seed=42,
        controller_halfsize=.003, controller_anchor="semantic",
        controller_vscale=None, start_frame=0, representation="tet_volume",
        controller_mode=cli.controller_mode,
        observation_readout=cli.observation_readout,
        grid_damping=math.sqrt(.98), floor_surface="frictional",
        floor_friction=0.9,
    )
    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = cli.case_dir.name
    cfg.train_frame = TRAIN_FRAME_COUNT
    set_all_seeds(args.seed)
    trainer = InvPhyTrainerWarpMPM(
        data_path=str(case_path), base_dir=str(cli.output_dir),
        train_frame=TRAIN_FRAME_COUNT, save_artifacts=False,
        mpm_kwargs={"grid_size": args.grid_size, "grid_lim": args.grid_lim,
                    "frame_dt": args.frame_dt, "sim_dt": SIM_DT},
        mpm_shift=args.sim_shift,
    )
    recorded_controller_inputs = None
    if args.controller_mode == "grid_cubes":
        configure_controllers(trainer, labels, args)
    else:
        recorded_controller_inputs = prepare_recorded_finger_patches(
            cli.source_run, source["source_steps"], source["rigid_poses"],
            len(labels), args.sim_shift, args.frame_dt, SIM_DT, cfg.device)

    summary = {
        "config": {
            "case_dir": str(cli.case_dir), "grid_size": args.grid_size,
            "grid_lim": args.grid_lim, "sim_shift": list(args.sim_shift),
            "frame_dt": args.frame_dt, "seed": args.seed,
            "controller_mode": args.controller_mode,
            "observation_readout": args.observation_readout,
            "observation_field": str(cli.observation_field),
            "floor_surface": args.floor_surface,
            "floor_friction": args.floor_friction,
        },
        "source_tracks": str(cli.source_tracks),
        "source_run": str(cli.source_run),
        "source_initial_state": str(cli.source_run / "initial_state.npz"),
        "source_selected_state": str(cli.source_run / "states" / "000000.npz"),
        "source_snap_bindings": str(cli.source_run / "snap_bindings.json"),
        "controller_pose_source": f"{cli.source_tracks}:rigid_poses",
        "source0_initialization": "source0 positions and velocities; no fitted initial state",
        "observation_prediction": ("source0 fixed material-face corners transformed by actual "
            "initialized owner center and completed endpoint F_trial, then proper triangle "
            "Kabsch transfer" if material_face_inputs is not None else
            "independently advected grid markers"),
        "legacy_grid_markers": "retained unchanged as same-run diagnostic observations",
        "observation_correspondence": ("privileged fixed synthetic source face correspondence "
            "shared by both fit arms; no future object targets used for readout"),
        "holdout_disclosure": ("source135--149 does not select fitted parameters but was "
                               "previously inspected during model debugging"),
        "material_prior_disclosure": "source material asset supplied the semantic prior",
        "representation": "tet_volume",
        "actual_sim_dt": SIM_DT,
        "grid_velocity_damping": args.grid_damping,
        "source_step_ranges": {
            "initialization": [0, 0], "train": [1, 134],
            "holdout": [135, 149], "full_replay": [0, 149],
        },
        "observation_count": int(len(labels)),
        "material_tet_count": int(len(tet_labels)),
        "body_tet_volume_fraction": body_volume_fraction,
        "semantic_prior_hardware_body_ratio": 440.0,
        "semantic_seed_normalization": "source0 tet-volume arithmetic mean 8000 Pa",
        "seed_E_pa": {arm: values.tolist() for arm, values in seed_E.items()},
        "method": "forward numerical optimization of corrected EMPM",
        "parameterization": "two free direct log-E values: body and all hardware, Pa",
        "loss": "pooled visible source states 1--134 squared Euclidean distance, world mm^2",
        "selection": "lowest finite evaluated training loss, including seed; no holdout or full replay selection",
        "optimizer": {
            "method": "Nelder-Mead", "maxfev": MAX_EVALUATIONS,
            "initial_simplex_log_offsets": [[0.0, 0.0], [.5, 0.0], [0.0, .5]],
            "xatol": 1e-10, "fatol": 1e-10,
        },
        "progress": {"active_arm": None, "completed_arms": [], "arms": {}},
        "arms": {},
    }
    if recorded_controller_inputs is None:
        summary["config"].update({
            "controller_halfsize": args.controller_halfsize,
            "controller_anchor": args.controller_anchor,
            "controller_vscale": args.controller_vscale,
        })
    else:
        summary.update({
            "controller_pose_convention": "world xyz+wxyz; wxyz reordered to xyzw for SciPy",
            "controller_interpolation": "linear translation and shortest-arc SciPy Slerp at integer substep ticks",
            "controller_patch_definitions": recorded_controller_inputs["patches"],
            "controller_offset_units": "world metres; simulated offsets divide by 5",
            "controller_approximation": ("kinematic hard attachment to four source finger links; "
                                         "source snaps had finite stiffness and native contact"),
            "controller_contact_association": ("privileged synthetic source binding metadata "
                                               "for 13 vertex/link associations, shared by both "
                                               "potential fit arms; no future object targets"),
            "grid_cuboid_count": 0, "grid_collider_count": 1,
        })
        if args.controller_mode == "recorded_finger_grid_centroids":
            summary["controller_approximation"] = (
                "mass-weighted patch-centroid grid projection, actual simulated-state "
                "capture at 39/40; hard centroids approximate finite source snaps/native contact")
            summary["controller_grid_projection"] = (
                "joint exact float64 solve with diagonal equilibration after frictional floor; "
                "correction only above floor at grid mass >1e-15; no particle writes; "
                "ordinary marker advection; forward-only optimization")
    controller_evidence = {}
    readout_failures = {}
    summary_path = cli.output_dir / "summary.json"
    save_summary(summary_path, summary)
    fields = {f"{arm}_initial_pa": material_field(np.log(values), labels)
              for arm, values in seed_E.items()}
    np.savez_compressed(cli.output_dir / "moduli.npz", **fields)

    with (cli.output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        stream.flush()

        for arm in ("default", "spatial"):
            set_all_seeds(args.seed)
            arm_started = time.perf_counter()
            q0 = np.log(seed_E[arm])
            simplex = np.stack((q0, q0 + [.5, 0.0], q0 + [0.0, .5]))
            calls = 0
            best_loss = float("inf")
            best_call = None
            best_q = None
            seed_loss = None
            summary["progress"]["active_arm"] = arm
            summary["progress"]["arms"][arm] = {
                "calls": 0, "current_best_train_mse_mm2": None,
                "current_best_call": None, "elapsed_arm_wall_s": 0.0,
            }
            save_summary(summary_path, summary)

            def objective(q):
                nonlocal calls, best_loss, best_call, best_q, seed_loss
                candidate_started = time.perf_counter()
                E_pa = material_field(q, labels)
                full_prediction, initialized = fixed_rollout(
                    trainer, E_pa, velocity, args, TRAIN_FRAME_COUNT, 0,
                    geometry, labels, recorded_controller_inputs, material_face_inputs)
                diagnostics = initialized.pop("controller_diagnostics", None)
                readout = initialized.pop("observation_readout_result")
                readout_status = readout["summary"]["status"]
                if diagnostics is not None and args.controller_mode == "recorded_finger_grid_centroids":
                    evidence_key = f"{arm}_call{calls + 1:03d}"
                    for name, value in diagnostics.items():
                        if name != "summary":
                            controller_evidence[f"{evidence_key}_{name}"] = value
                    controller_evidence[f"{evidence_key}_summary_json"] = np.array(
                        json.dumps(diagnostics["summary"]))
                    np.savez_compressed(cli.output_dir / "controller_evidence.npz",
                                        **controller_evidence)
                raw_loss = float("nan")
                raw_status = "incomplete_raw_trajectory"
                if len(full_prediction) == TRAIN_FRAME_COUNT:
                    raw_distance2 = ((full_prediction[1:135, :len(labels)] - target[1:135])
                                     ** 2).sum(axis=-1)
                    raw_loss = float(1e6 * raw_distance2[visible[1:135]].mean())
                    raw_status = "ok" if np.isfinite(raw_loss) else "nonfinite_raw_loss"
                if readout_status == "numerical_failure":
                    failure = readout["summary"]
                    evidence_key = f"{arm}_call{calls + 1:03d}"
                    frame = failure["first_failed_frame"]
                    affected = np.asarray(failure["affected_observation_ids"], dtype=np.int64)
                    full_ids = material_face_inputs["material_face_unique_owner_full_ids"]
                    readout_failures[f"{evidence_key}_summary_json"] = np.array(
                        json.dumps(failure))
                    readout_failures[f"{evidence_key}_owner_x_world_m"] = (
                        full_prediction[frame, full_ids].copy())
                    readout_failures[f"{evidence_key}_owner_F_trial"] = (
                        readout["owner_F_trial"][frame].copy())
                    readout_failures[f"{evidence_key}_owner_initial_world_m"] = (
                        readout["owner_initial_world_m"].copy())
                    readout_failures[f"{evidence_key}_affected_observation_ids"] = affected
                    readout_failures[f"{evidence_key}_affected_face_vertex_ids"] = (
                        material_face_inputs["material_face_vertex_ids"][affected])
                    readout_failures[f"{evidence_key}_affected_rest_vertices_world_m"] = (
                        material_face_inputs["material_face_rest_vertices_world_m"][affected])
                    readout_failures[f"{evidence_key}_affected_rest_observations_world_m"] = (
                        material_face_inputs[
                            "material_face_rest_observations_world_m"][affected])
                    np.savez_compressed(cli.output_dir / "readout_failures.npz",
                                        **readout_failures)
                    summary.setdefault("readout_failures", {})[evidence_key] = failure
                controller_failed = (diagnostics is not None and
                        (diagnostics["summary"].get("status") == "numerical_failure" or
                         diagnostics["summary"].get(
                             "max_active_grid_centroid_velocity_residual_world_mps", 0.0)
                         >= 1e-5))
                if controller_failed:
                    loss = float("inf")
                    print(f"WARNING {arm} call={calls + 1}: unsatisfied grid projection",
                          flush=True)
                elif readout_status == "numerical_failure" or (
                        readout["prediction_world_m"] is None or
                        len(readout["prediction_world_m"]) != TRAIN_FRAME_COUNT):
                    loss = float("inf")
                else:
                    distance2 = ((readout["prediction_world_m"][1:135] - target[1:135])
                                 ** 2).sum(axis=-1)
                    loss = float(1e6 * distance2[visible[1:135]].mean())
                    if not np.isfinite(loss):
                        readout_status = "nonfinite_training_loss"
                calls += 1
                if calls == 1:
                    seed_loss = loss
                if np.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_call = calls
                    best_q = q.copy()
                if not np.isfinite(loss):
                    print(f"WARNING {arm} call={calls}: nonfinite training loss",
                          flush=True)
                    loss = float("inf")
                elapsed = time.perf_counter() - arm_started
                candidate_duration = time.perf_counter() - candidate_started
                E = np.exp(q)
                writer.writerow({
                    "arm": arm, "call": calls,
                    "log_body_E_pa": float(q[0]),
                    "log_hardware_E_pa": float(q[1]),
                    "body_E_pa": float(E[0]),
                    "hardware_E_pa": float(E[1]),
                    "train_mse_mm2": loss,
                    "grid_marker_train_mse_mm2": raw_loss,
                    "grid_marker_status": raw_status,
                    "readout_status": readout_status,
                    "best_train_mse_mm2": best_loss,
                    "best_call": best_call,
                    "candidate_duration_s": candidate_duration,
                    "elapsed_arm_wall_s": elapsed,
                })
                stream.flush()
                summary["progress"]["arms"][arm] = {
                    "calls": calls,
                    "current_best_train_mse_mm2": (
                        best_loss if np.isfinite(best_loss) else None),
                    "current_best_call": best_call,
                    "elapsed_arm_wall_s": elapsed,
                }
                save_summary(summary_path, summary)
                print(f"{arm} call={calls} train_mse_mm2={loss:.6f} "
                      f"best={best_loss:.6f} elapsed_s={elapsed:.1f}", flush=True)
                return loss

            result = minimize(
                objective, q0, method="Nelder-Mead",
                options={"initial_simplex": simplex, "maxfev": MAX_EVALUATIONS,
                         "xatol": 1e-10, "fatol": 1e-10},
            )
            if best_q is None:
                summary["arms"][arm] = {
                    "status": "no_finite_candidate", "nfev": int(result.nfev),
                    "optimizer_message": str(result.message),
                    "wall_time_s": time.perf_counter() - arm_started,
                }
                summary["progress"]["active_arm"] = None
                save_summary(summary_path, summary)
                print(f"WARNING {arm}: no finite candidate; controller evidence saved",
                      flush=True)
                return
            selected_E = np.exp(best_q)
            fields[f"{arm}_eval{MAX_EVALUATIONS}_pa"] = material_field(best_q, labels)
            np.savez_compressed(cli.output_dir / "moduli.npz", **fields)
            arm_wall_time = time.perf_counter() - arm_started
            summary["arms"][arm] = {
                "seed_call": 1, "seed_q": q0.tolist(),
                "seed_E_pa": seed_E[arm].tolist(),
                "seed_train_mse_mm2": seed_loss,
                "selected_call": best_call,
                "selected_q": best_q.tolist(),
                "selected_E_pa": selected_E.tolist(),
                "selected_train_mse_mm2": best_loss,
                "nfev": int(result.nfev),
                "converged_before_maxfev": bool(result.nfev < MAX_EVALUATIONS),
                "optimizer_message": str(result.message),
                "wall_time_s": arm_wall_time,
            }
            summary["progress"]["active_arm"] = None
            summary["progress"]["completed_arms"].append(arm)
            summary["progress"]["arms"][arm]["elapsed_arm_wall_s"] = arm_wall_time
            save_summary(summary_path, summary)


if __name__ == "__main__":
    main()
