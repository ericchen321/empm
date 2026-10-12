"""Replay a completed Nailoong fit with fresh MPM states (run from repo root)."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import cfg
from scripts.train_test.run_nailoong_material_init import (
    configure_controllers, rollout, temporal_train_loss,
)
from scripts.utils import set_all_seeds


def rms_mm(delta, mask=None):
    squared = np.sum(np.asarray(delta) ** 2, axis=-1)
    if mask is not None:
        squared = squared[mask]
    return float(np.sqrt(squared.mean()) * 1e3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--evaluations", type=int, nargs="+",
                        help="Checkpoint evaluation budgets (default: final budget)")
    cli = parser.parse_args()
    output_dir = cli.output_dir or cli.run_dir / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (cli.run_dir / "summary.json").open() as f:
        summary = json.load(f)
    args = SimpleNamespace(**summary["config"])
    args.case_dir = Path(args.case_dir)
    args.field = Path(args.field)
    args.grid_damping = getattr(args, "grid_damping", summary["grid_velocity_damping"])
    budgets = cli.evaluations or [args.max_evaluations]
    train_len = args.train_end - args.start_frame

    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = args.case_dir.name
    cfg.train_frame = args.train_end
    set_all_seeds(args.seed)
    labels = np.load(args.field)["part_labels"]
    trainer = InvPhyTrainerWarpMPM(
        data_path=str(args.case_dir / "final_data.pkl"), base_dir=str(output_dir),
        train_frame=args.train_end, save_artifacts=False,
        mpm_kwargs={"grid_size": args.grid_size, "grid_lim": args.grid_lim,
                    "frame_dt": args.frame_dt, "sim_dt": summary["actual_sim_dt"]},
        mpm_shift=args.sim_shift,
    )
    configure_controllers(trainer, labels, args)
    moduli = np.load(cli.run_dir / "moduli.npz")
    predictions = np.load(cli.run_dir / "predictions.npz")
    with (cli.run_dir / "search.csv").open(newline="") as f:
        search = list(csv.DictReader(f))

    result = {"source_run": str(cli.run_dir), "evaluation_budgets": budgets,
              "repeats": cli.repeats, "actual_sim_dt": summary["actual_sim_dt"],
              "grid_damping_per_substep": args.grid_damping, "budgets": {}}
    target = trainer.object_points[args.train_end:args.eval_end].cpu().numpy()
    visible = trainer.object_visibilities[args.train_end:args.eval_end].cpu().numpy()

    for budget in budgets:
        budget_result = {"arms": {}}
        result["budgets"][str(budget)] = budget_result
        for arm in args.arms:
            E_pa = moduli[f"{arm}_eval{budget}_pa"]
            stored = predictions[f"{arm}_eval{budget}_world_m"]
            recorded = next(r for r in search if r["arm"] == arm
                            and int(r["evaluation"]) == budget)
            arm_result = {
                "recorded_best_train_mse_mm2": float(recorded["best_train_mse_mm2"]),
                "stored_checkpoint_train_mse_mm2": temporal_train_loss(
                    torch.as_tensor(stored[:train_len], device=cfg.device), trainer, args),
                "stored_checkpoint_heldout_rmse_mm": rms_mm(
                    stored[train_len:] - target, visible),
                "runs": [],
            }
            budget_result["arms"][arm] = arm_result
            for repeat in range(cli.repeats):
                set_all_seeds(args.seed)
                started = time.perf_counter()
                train = rollout(trainer, E_pa, args, args.train_end)
                full = rollout(trainer, E_pa, args, args.eval_end)
                train_np = train.cpu().numpy()
                full_np = full.cpu().numpy()
                prefix_delta = full_np[:train_len] - stored[:train_len]
                run = {
                    "repeat": repeat + 1,
                    "fresh_train_mse_mm2": temporal_train_loss(train, trainer, args),
                    "fresh_full_prefix_train_mse_mm2": temporal_train_loss(
                        full[:train_len], trainer, args),
                    "fresh_heldout_rmse_mm": rms_mm(full_np[train_len:] - target, visible),
                    "fresh_train_vs_full_prefix_rmse_mm": rms_mm(train_np - full_np[:train_len]),
                    "fresh_full_vs_stored_train_prefix_rmse_mm": rms_mm(prefix_delta),
                    "fresh_full_vs_stored_train_prefix_max_mm": float(
                        np.linalg.norm(prefix_delta, axis=-1).max() * 1e3),
                    "fresh_full_vs_stored_all_rmse_mm": rms_mm(full_np - stored),
                    "elapsed_wall_time_s": time.perf_counter() - started,
                }
                arm_result["runs"].append(run)
                with (output_dir / "replay.json").open("w") as f:
                    json.dump(result, f, indent=2)
                print(f"eval={budget} {arm} repeat={repeat + 1} "
                      f"train_mse={run['fresh_train_mse_mm2']:.6f} "
                      f"heldout_rmse_mm={run['fresh_heldout_rmse_mm']:.6f}", flush=True)


if __name__ == "__main__":
    main()
