"""Controlled Nailoong material initialization with forward-only system ID.

Every arm fits the same body/hardware log-modulus pair. Only the initial pair
changes; contact, solver, training frames, objective, and search budget match.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.engine.utils import initial_param_guess, inv_transform_points, transform_points
from phys_sim.utils import cfg
from scripts.utils import set_all_seeds


PART_NAMES = {0: "body", 1: "connector", 2: "ring", 3: "clasp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Path("outputs/double_nailoong_grippers_20261011"))
    parser.add_argument("--field", type=Path, default=Path("outputs/nailoong_material_init/field.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nailoong_material_init/experiment"))
    parser.add_argument("--arms", nargs="+", choices=("default", "spatial", "mean", "reversed"),
                        default=("default", "spatial", "mean", "reversed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-frame", type=int, default=40)
    parser.add_argument("--train-end", type=int, default=44, help="Exclusive train frame index")
    parser.add_argument("--eval-end", type=int, default=48, help="Exclusive evaluation frame index")
    parser.add_argument("--max-evaluations", type=int, default=40,
                        help="Nelder-Mead training-objective evaluation budget")
    parser.add_argument("--iterations", type=int, dest="max_evaluations",
                        help="Alias for --max-evaluations (0 runs initial rollouts only)")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=(10, 20, 40),
                        help="Training-evaluation budgets for held-out metrics")
    parser.add_argument("--vlm-scale", type=float, default=1.0,
                        help="Multiplier applied to the source VLM field; 1 is raw VLM")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--grid-lim", type=float, default=.064)
    parser.add_argument("--sim-shift", type=float, nargs=3, default=(.54, .47, .496))
    parser.add_argument("--frame-dt", type=float, default=1 / 30)
    parser.add_argument("--sim-dt", type=float, default=1e-3)
    parser.add_argument("--grid-damping", type=float, default=.98,
                        help="Grid velocity multiplier applied at each simulation substep")
    parser.add_argument("--controller-halfsize", type=float, default=.003)
    parser.add_argument("--controller-anchor", choices=("semantic", "native"), default="semantic")
    parser.add_argument("--controller-vscale", type=float, default=None,
                        help="Observed per-frame delta to solver velocity; default 1/frame_dt")
    parser.add_argument("--active-start", type=int, default=40)
    return parser.parse_args()


def configure_controllers(trainer, labels, args):
    trainer.ctrl_param["csize"] = [args.controller_halfsize] * trainer.n_ctrl_parts
    speed = args.controller_vscale if args.controller_vscale is not None else 1 / args.frame_dt
    trainer.ctrl_param["vscale"] = [[speed] * 3 for _ in range(trainer.n_ctrl_parts)]
    if args.controller_anchor == "semantic":
        for i, part_label in enumerate((2, 0)):
            ctrl = trainer.ctrl_points[0, trainer.ctrl_param["ctrl_mask"] == i]
            obj_indices = torch.nonzero(torch.as_tensor(labels == part_label, device=ctrl.device)).flatten()
            d = torch.cdist(ctrl, trainer.obj_points[0, obj_indices])
            obj_idx = obj_indices[torch.argmin(d.min(dim=0).values)]
            trainer.ctrl_param["handler_pos"][i] = trainer.obj_points[0, obj_idx].cpu().numpy()
        trainer.ctrl_param["ctrl_index"] = [-1] * trainer.n_ctrl_parts


def make_mpm(trainer, E_pa, args, diff_sim):
    params = initial_param_guess(name=trainer.exp_name)
    params["E"] = torch.as_tensor(E_pa / 1e6, dtype=torch.float32, device=cfg.device)
    mpm = trainer.initialize_mpm(params, diff_sim=diff_sim, update_ctrl=True)
    mpm.mpm_model.grid_v_damping_scale = getattr(args, "grid_damping", .98)

    x = transform_points(trainer.object_points[args.start_frame].clone(), name=trainer.exp_name)
    x -= torch.as_tensor(trainer.mpm_shift, dtype=x.dtype, device=x.device)
    floor = .5 - trainer.mpm_shift[2]
    if x[:, 2].min() < floor + 1e-4:
        x[:, 2] += .001
    mpm.x_t = x
    mpm.mpm_state.reset_state(x, None, mpm.v_t, device=mpm.device, requires_grad=diff_sim)

    original_callback = mpm.frame_callback
    mpm.frame_callback = lambda wrapper, frame: original_callback(wrapper, frame + args.start_frame)
    for i in range(trainer.n_ctrl_parts):
        anchor = np.asarray(trainer.ctrl_param["handler_pos"][i])
        mask = trainer.ctrl_param["ctrl_mask"] == i
        displacement = (trainer.ctrl_points[args.start_frame, mask].mean(0)
                        - trainer.ctrl_points[0, mask].mean(0)).cpu().numpy()
        point = anchor + displacement - trainer.mpm_shift
        mpm.mpm_solver.collider_params[i + 1].point = wp.vec3(*point)
    mpm.mpm_solver.time = 0.0
    return mpm


def material_field(log_parameters, labels):
    body_fraction = np.mean(labels == 0)
    hardware_fraction = 1 - body_fraction
    log_body = log_parameters[0] - hardware_fraction * log_parameters[1]
    log_hardware = log_parameters[0] + body_fraction * log_parameters[1]
    return np.where(labels == 0, np.exp(log_body), np.exp(log_hardware)).astype(np.float32)


def material_values(log_parameters, labels):
    field = material_field(log_parameters, labels)
    return float(field[labels == 0][0]), float(field[labels != 0][0])


def temporal_train_loss(prediction, trainer, args):
    target = trainer.object_points[args.start_frame + 1:args.train_end]
    visible = trainer.object_visibilities[args.start_frame + 1:args.train_end]
    squared_mm = ((prediction[1:] - target) ** 2).sum(-1) * 1e6
    per_frame = (squared_mm * visible).sum(-1) / visible.sum(-1)
    return float(per_frame.mean().item())


def metric_rows(arm, evaluation, prediction, trainer, labels, args):
    target = trainer.object_points[args.start_frame:args.eval_end]
    visible = trainer.object_visibilities[args.start_frame:args.eval_end]
    distance_mm = ((prediction - target).norm(dim=-1) * 1e3).detach().cpu().numpy()
    visible = visible.cpu().numpy()
    frame_ids = np.arange(args.start_frame, args.eval_end)
    splits = {
        "train": (frame_ids > args.start_frame) & (frame_ids < args.train_end),
        "active_train": (frame_ids >= max(args.active_start, args.start_frame + 1))
                        & (frame_ids < args.train_end),
        "heldout": (frame_ids >= args.train_end),
    }
    rows = []
    for split, frames in splits.items():
        for part, part_name in [(None, "all"), *PART_NAMES.items()]:
            mask = visible & frames[:, None]
            if part is not None:
                mask &= labels[None, :] == part
            values = distance_mm[mask]
            rows.append({"arm": arm, "evaluation": evaluation, "split": split,
                         "part": part_name, "count": int(values.size),
                         "rmse_mm": float(np.sqrt(np.mean(values ** 2))),
                         "mean_error_mm": float(values.mean())})
    return rows


def rollout(trainer, E_pa, args, end_frame):
    mpm = make_mpm(trainer, E_pa, args, diff_sim=False)
    vertices = mpm.advance_to(end_frame - args.start_frame - 1)
    shifted = vertices + torch.as_tensor(trainer.mpm_shift, dtype=vertices.dtype, device=vertices.device)
    return inv_transform_points(shifted, name=trainer.exp_name)


def save_results(output_dir, rows, history, result, parameter_arrays):
    np.savez_compressed(output_dir / "moduli.npz", **parameter_arrays)
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "search.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("arm", "evaluation", "log_mean_pa",
                            "log_contrast", "body_E_pa", "hardware_E_pa",
                            "train_mse_mm2", "best_train_mse_mm2", "wall_time_s"))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = args.case_dir.name
    cfg.train_frame = args.train_end
    set_all_seeds(args.seed)
    field = np.load(args.field)
    raw_E = field["youngs_modulus_pa"].astype(np.float32)
    labels = field["part_labels"].astype(np.int64)
    default_E = initial_param_guess(name=cfg.case_name)["E"] * 1e6
    spatial_E = raw_E * args.vlm_scale
    body_fraction = float(np.mean(labels == 0))
    hardware_fraction = 1 - body_fraction
    spatial_body = float(spatial_E[labels == 0][0])
    spatial_hardware = float(spatial_E[labels != 0][0])
    spatial_contrast = math.log(spatial_hardware / spatial_body)
    spatial_log_mean = (body_fraction * math.log(spatial_body)
                        + hardware_fraction * math.log(spatial_hardware))
    initial_parameters = {
        "default": np.array([math.log(default_E), 0.0]),
        "spatial": np.array([spatial_log_mean, spatial_contrast]),
        "mean": np.array([spatial_log_mean, 0.0]),
        "reversed": np.array([spatial_log_mean, -spatial_contrast]),
    }
    n_substeps = math.ceil(args.frame_dt / args.sim_dt)
    actual_sim_dt = args.frame_dt / n_substeps

    trainer = InvPhyTrainerWarpMPM(
        data_path=str(args.case_dir / "final_data.pkl"), base_dir=str(args.output_dir),
        train_frame=args.train_end, save_artifacts=False,
        mpm_kwargs={"grid_size": args.grid_size, "grid_lim": args.grid_lim,
                    "frame_dt": args.frame_dt, "sim_dt": actual_sim_dt},
        mpm_shift=args.sim_shift,
    )
    assert np.array_equal(field["structure_points_m"], trainer.object_points[0].cpu().numpy())
    configure_controllers(trainer, labels, args)
    checkpoint_steps = {n for n in args.checkpoints if 0 < n <= args.max_evaluations}
    checkpoint_steps.add(args.max_evaluations)
    rows = []
    history = []
    result = {"config": {**vars(args), "case_dir": str(args.case_dir), "field": str(args.field),
                         "output_dir": str(args.output_dir)},
              "default_E_pa": default_E, "raw_VLM_E_pa": sorted(np.unique(raw_E).tolist()),
              "effective_VLM_E_pa": sorted(np.unique(spatial_E).tolist()),
              "body_fraction": body_fraction, "hardware_fraction": hardware_fraction,
              "initial_log_parameters": {k: v.tolist() for k, v in initial_parameters.items()},
              "n_particles": len(raw_E), "n_substeps_per_frame": n_substeps,
              "actual_sim_dt": actual_sim_dt, "dx": args.grid_lim / args.grid_size,
              "material_type": "jelly", "poisson_ratio": 0.3,
              "density_kg_m3": 2000.0, "particle_volume": 1e-5,
              "gravity_sim_m_s2": [0.0, 0.0, -9.8],
              "grid_velocity_damping": args.grid_damping,
              "controller_positions_transformed": [np.asarray(x).tolist()
                                                   for x in trainer.ctrl_param["handler_pos"]],
              "controller_vscale": trainer.ctrl_param["vscale"],
              "field_provenance": str(args.field.parent / "provenance.json"),
              "numeric_search": {"method": "Nelder-Mead", "parameters":
                                 "weighted log mean Pa and log hardware/body contrast",
                                 "initial_simplex_log_increment": 0.5,
                                 "maxfev": args.max_evaluations,
                                 "checkpoint_selection": "best training loss only"},
              "adjoint_diagnostic": {"configuration": "uniform 8000 Pa, start frame 40, one transition, "
                                                 "local grid32/.064, dt 0.0001996008",
                                     "before_repair_warp_directional_derivative": -0.00010961364750983194,
                                     "after_repair_warp_directional_derivative": -0.0052042510360,
                                     "centered_forward_difference": -0.005204556509852409,
                                     "after_repair_relative_error_percent": 0.00587,
                                     "verification": "docs/mpm_gradient_verification.md (22 directional checks)",
                                     "search_rationale": "Two-parameter forward search supports the full horizon "
                                                         "without storing an adjoint rollout"},
              "arm_wall_time_s": {},
              "loss": "mean over training frames of mean visible tracked squared distance in mm^2",
              "metric": "visible tracked-point Euclidean RMSE in original world millimeters",
              "arm_best_train_mse_mm2": {}, "arm_optimizer": {}}
    parameter_arrays = {}
    prediction_arrays = {}

    for arm in args.arms:
        arm_start_time = time.perf_counter()
        set_all_seeds(args.seed)
        start_parameters = initial_parameters[arm]
        initial_pa = material_field(start_parameters, labels)
        initial_points = rollout(trainer, initial_pa, args, args.eval_end)
        initial_loss = temporal_train_loss(
            initial_points[:args.train_end - args.start_frame], trainer, args)
        best_parameters = start_parameters.copy()
        best_loss = initial_loss
        rows.extend(metric_rows(arm, 0, initial_points, trainer, labels, args))
        parameter_arrays[f"{arm}_initial_pa"] = initial_pa
        prediction_arrays[f"{arm}_eval0_world_m"] = initial_points.cpu().numpy()
        np.savez_compressed(args.output_dir / "predictions.npz", **prediction_arrays)
        history.append({"arm": arm, "evaluation": 0,
                        "log_mean_pa": float(start_parameters[0]),
                        "log_contrast": float(start_parameters[1]),
                        "body_E_pa": material_values(start_parameters, labels)[0],
                        "hardware_E_pa": material_values(start_parameters, labels)[1],
                        "train_mse_mm2": initial_loss,
                        "best_train_mse_mm2": best_loss,
                        "wall_time_s": time.perf_counter() - arm_start_time})
        result["arm_best_train_mse_mm2"][arm] = best_loss
        result["arm_wall_time_s"][arm] = time.perf_counter() - arm_start_time
        save_results(args.output_dir, rows, history, result, parameter_arrays)

        evaluation = 0
        recorded = {0}

        def record_best(budget):
            current_pa = material_field(best_parameters, labels)
            prediction = rollout(trainer, current_pa, args, args.eval_end)
            rows.extend(metric_rows(arm, budget, prediction, trainer, labels, args))
            parameter_arrays[f"{arm}_eval{budget}_pa"] = current_pa
            prediction_arrays[f"{arm}_eval{budget}_world_m"] = prediction.cpu().numpy()
            np.savez_compressed(args.output_dir / "predictions.npz", **prediction_arrays)
            result["arm_best_train_mse_mm2"][arm] = best_loss
            result["arm_wall_time_s"][arm] = time.perf_counter() - arm_start_time
            save_results(args.output_dir, rows, history, result, parameter_arrays)
            recorded.add(budget)

        def objective(log_parameters):
            nonlocal evaluation, best_loss, best_parameters
            E_pa = material_field(log_parameters, labels)
            prediction = rollout(trainer, E_pa, args, args.train_end)
            loss = temporal_train_loss(prediction, trainer, args)
            evaluation += 1
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_parameters = log_parameters.copy()
            if not np.isfinite(loss):
                print(f"WARNING {arm} evaluation={evaluation}: nonfinite training loss", flush=True)
                loss = float("inf")
            body_E, hardware_E = material_values(log_parameters, labels)
            history.append({"arm": arm, "evaluation": evaluation,
                            "log_mean_pa": float(log_parameters[0]),
                            "log_contrast": float(log_parameters[1]),
                            "body_E_pa": body_E, "hardware_E_pa": hardware_E,
                            "train_mse_mm2": loss, "best_train_mse_mm2": best_loss,
                            "wall_time_s": time.perf_counter() - arm_start_time})
            result["arm_best_train_mse_mm2"][arm] = best_loss
            result["arm_wall_time_s"][arm] = time.perf_counter() - arm_start_time
            print(f"{arm} evaluation={evaluation} train_mse_mm2={loss:.6f} "
                  f"best={best_loss:.6f}", flush=True)
            if evaluation in checkpoint_steps:
                record_best(evaluation)
            else:
                save_results(args.output_dir, rows, history, result, parameter_arrays)
            return loss

        if args.max_evaluations > 0:
            simplex = np.stack((start_parameters,
                                start_parameters + np.array([.5, 0.]),
                                start_parameters + np.array([0., .5])))
            optimizer_result = minimize(
                objective, start_parameters, method="Nelder-Mead",
                options={"initial_simplex": simplex, "maxfev": args.max_evaluations,
                         "xatol": 1e-10, "fatol": 1e-10})
            result["arm_optimizer"][arm] = {"nfev": int(optimizer_result.nfev),
                                             "message": str(optimizer_result.message)}
            if evaluation not in recorded:
                record_best(evaluation)
        else:
            result["arm_optimizer"][arm] = {"nfev": 0, "message": "initial rollout only"}
        result["arm_best_train_mse_mm2"][arm] = best_loss
        result["arm_wall_time_s"][arm] = time.perf_counter() - arm_start_time
        save_results(args.output_dir, rows, history, result, parameter_arrays)


if __name__ == "__main__":
    main()
