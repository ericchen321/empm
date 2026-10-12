"""Compare Nailoong material-initialization runs and plot saved trajectories."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
COLORS = {"default": "#1f77b4", "spatial": "#d62728",
          "mean": "#2ca02c", "reversed": "#9467bd"}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_files(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_run(run_dir):
    with (run_dir / "summary.json").open() as f:
        summary = json.load(f)
    config = summary["config"]
    case_path = Path(config["case_dir"])
    if not case_path.is_absolute():
        case_path = REPO_ROOT / case_path
    with (case_path / "final_data.pkl").open("rb") as f:
        case = pickle.load(f)
    field_path = Path(config["field"])
    if not field_path.is_absolute():
        field_path = REPO_ROOT / field_path
    return {"name": run_dir.name, "case": case_path.name, "summary": summary,
            "search": read_csv(run_dir / "search.csv"),
            "metrics": read_csv(run_dir / "metrics.csv"),
            "predictions": np.load(run_dir / "predictions.npz"),
            "moduli": np.load(run_dir / "moduli.npz"),
            "labels": np.load(field_path)["part_labels"], "data": case}


def comparison_rows(run):
    search = {(r["arm"], int(r["evaluation"])): r for r in run["search"]}
    metrics = {(r["arm"], int(r["evaluation"]), r["split"], r["part"]): r
               for r in run["metrics"]}
    keys = sorted({(r["arm"], int(r["evaluation"])) for r in run["metrics"]},
                  key=lambda x: (x[0], x[1]))
    rows = []
    for arm, budget in keys:
        objective = search[(arm, budget)]
        value = lambda split, part: float(metrics[(arm, budget, split, part)]["rmse_mm"])
        field_key = f"{arm}_initial_pa" if budget == 0 else f"{arm}_eval{budget}_pa"
        moduli = run["moduli"][field_key]
        config = run["summary"]["config"]
        start, end = config["start_frame"], config["train_end"]
        prediction = run["predictions"][f"{arm}_eval{budget}_world_m"]
        visible = run["data"]["object_visibilities"][start + 1:end].astype(bool)
        squared_mm = np.sum((prediction[1:end - start]
                             - run["data"]["object_points"][start + 1:end]) ** 2, axis=-1) * 1e6
        replay_mse = float(((squared_mm * visible).sum(-1) / visible.sum(-1)).mean())
        recorded_mse = float(objective["best_train_mse_mm2"])
        rows.append({"scenario": run["name"], "case": run["case"],
                     "arm": arm, "evaluation_budget": budget,
                     "recorded_incumbent_train_objective_root_mm": float(np.sqrt(recorded_mse)),
                     "saved_checkpoint_train_objective_root_mm": float(np.sqrt(replay_mse)),
                     "saved_vs_recorded_train_mse_relative_difference":
                         (replay_mse - recorded_mse) / recorded_mse,
                     "train_pooled_rmse_mm": value("train", "all"),
                     "active_train_pooled_rmse_mm": value("active_train", "all"),
                     "heldout_pooled_rmse_mm": value("heldout", "all"),
                     "heldout_body_rmse_mm": value("heldout", "body"),
                     "heldout_ring_rmse_mm": value("heldout", "ring"),
                     "best_body_E_pa": float(moduli[run["labels"] == 0][0]),
                     "best_hardware_E_pa": float(moduli[run["labels"] != 0][0])})
    return rows


def convergence_rows(run, comparisons):
    budget = 24
    default_initial = float(next(r["train_mse_mm2"] for r in run["search"]
                                 if r["arm"] == "default" and int(r["evaluation"]) == 0))
    threshold = .9 * default_initial
    rows = []
    for arm in run["summary"]["config"]["arms"]:
        checkpoints = [r for r in comparisons if r["scenario"] == run["name"]
                       and r["arm"] == arm]
        largest_gap = max(checkpoints, key=lambda r: abs(
            r["saved_vs_recorded_train_mse_relative_difference"]))
        search = sorted((r for r in run["search"] if r["arm"] == arm
                         and int(r["evaluation"]) <= budget),
                        key=lambda r: int(r["evaluation"]))
        evaluations = np.array([int(r["evaluation"]) for r in search])
        best_losses = np.array([float(r["best_train_mse_mm2"]) for r in search])
        # An improvement is available only after its evaluation completes.
        auc = np.sum(best_losses[:-1] * np.diff(evaluations))
        auc += (budget - evaluations[-1]) * best_losses[-1]
        first_hit = next((r for r in search
                          if float(r["best_train_mse_mm2"]) <= threshold), None)
        rows.append({"scenario": run["name"], "case": run["case"], "arm": arm,
                     "evaluation_budget": budget,
                     "default_initial_train_mse_mm2": default_initial,
                     "common_90pct_threshold_mse_mm2": threshold,
                     "recorded_initial_train_mse_mm2": float(search[0]["train_mse_mm2"]),
                     "recorded_incumbent_best_train_mse_mm2_at_budget": best_losses[-1],
                     "recorded_incumbent_normalized_mse_auc_0_24":
                         auc / (budget * default_initial),
                     "recorded_first_evaluation_to_threshold":
                         int(first_hit["evaluation"]) if first_hit else "",
                     "recorded_wall_time_to_threshold_s":
                         float(first_hit["wall_time_s"]) if first_hit else "",
                     "final_arm_wall_time_s":
                         float(run["summary"]["arm_wall_time_s"][arm]),
                     "largest_abs_saved_vs_recorded_train_mse_relative_gap":
                         abs(largest_gap["saved_vs_recorded_train_mse_relative_difference"]),
                     "largest_gap_evaluation_budget": largest_gap["evaluation_budget"],
                     "numerical_consistency_note":
                         "Recorded incumbent AUC and threshold are descriptive log values; "
                         "saved checkpoint objectives may differ and convergence is not "
                         "independently validated."})
    return rows


def frame_rows(run):
    config = run["summary"]["config"]
    start, cutoff = config["start_frame"], config["train_end"]
    target = run["data"]["object_points"]
    visible = run["data"]["object_visibilities"]
    rows = []
    for key in run["predictions"].files:
        arm, remainder = key.split("_eval", 1)
        budget = int(remainder.split("_world_m", 1)[0])
        prediction = run["predictions"][key]
        for offset, points in enumerate(prediction):
            frame = start + offset
            distance_sq_mm = np.sum((points - target[frame]) ** 2, axis=-1) * 1e6
            mask = visible[frame].astype(bool)
            split = "initial" if frame == start else "train" if frame < cutoff else "heldout"
            rows.append({"scenario": run["name"], "case": run["case"],
                         "arm": arm, "evaluation_budget": budget, "frame": frame,
                         "split": split, "visible_count": int(mask.sum()),
                         "rmse_mm": float(np.sqrt(distance_sq_mm[mask].mean()))})
    return rows


def optimization_figure(runs, comparisons, output_dir):
    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 3.5 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        left, middle, right = axes[row]
        run_checkpoints = [r for r in comparisons if r["scenario"] == run["name"]]
        for arm in run["summary"]["config"]["arms"]:
            search = [r for r in run["search"] if r["arm"] == arm]
            if not search:
                continue
            x = [int(r["evaluation"]) for r in search]
            y = [np.sqrt(float(r["best_train_mse_mm2"])) for r in search]
            left.plot(x, y, label=arm, color=COLORS[arm], linewidth=1.8)
            wall_minutes = [float(r["wall_time_s"]) / 60 for r in search]
            middle.plot(wall_minutes, y, label=arm, color=COLORS[arm], linewidth=1.8)
            checkpoints = [r for r in run_checkpoints if r["arm"] == arm]
            checkpoint_x = [r["evaluation_budget"] for r in checkpoints]
            checkpoint_y = [r["saved_checkpoint_train_objective_root_mm"]
                            for r in checkpoints]
            left.scatter(checkpoint_x, checkpoint_y, marker="X", s=45,
                         color=COLORS[arm], edgecolors="black", linewidths=.4, zorder=4)
            minutes_at_eval = {int(r["evaluation"]): float(r["wall_time_s"]) / 60
                               for r in search}
            middle.scatter([minutes_at_eval[x] for x in checkpoint_x], checkpoint_y,
                           marker="X", s=45, color=COLORS[arm], edgecolors="black",
                           linewidths=.4, zorder=4)
            right.plot([r["evaluation_budget"] for r in checkpoints],
                       [r["heldout_pooled_rmse_mm"] for r in checkpoints],
                       "o-", label=arm, color=COLORS[arm], linewidth=1.8, markersize=4)
        largest_gap = max(run_checkpoints, key=lambda r: abs(
            r["saved_checkpoint_train_objective_root_mm"]
            - r["recorded_incumbent_train_objective_root_mm"]))
        left.annotate(
            f"Largest recorded/saved gap: {largest_gap['arm']} @"
            f"{largest_gap['evaluation_budget']}\n"
            f"{largest_gap['recorded_incumbent_train_objective_root_mm']:.2f} / "
            f"{largest_gap['saved_checkpoint_train_objective_root_mm']:.2f} mm",
            xy=(largest_gap["evaluation_budget"],
                largest_gap["saved_checkpoint_train_objective_root_mm"]),
            xytext=(.02, .98), textcoords="axes fraction", va="top", fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "#555555"})
        left.plot([], [], "k-", label="recorded incumbent")
        left.scatter([], [], marker="X", color="white", edgecolors="black",
                     label="saved checkpoint")
        left.set_title(f"{run['name']} — training objective")
        middle.set_title(f"{run['name']} — training objective vs time")
        right.set_title(f"{run['name']} — saved held-out error")
        left.set_ylabel("√(mean per-frame visible MSE) (mm)")
        middle.set_ylabel("√(mean per-frame visible MSE) (mm)")
        right.set_ylabel("Pooled visible-point RMSE (mm)")
        left.set_xlabel("Forward objective evaluations")
        middle.set_xlabel("Arm elapsed wall time (min)")
        right.set_xlabel("Forward objective evaluations")
        left.set_yscale("log")
        middle.set_yscale("log")
        for ax in (left, middle, right):
            ax.grid(alpha=.25)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plot_files(fig, output_dir, "optimization_curves")


def budget8_figure(runs, comparisons, output_dir):
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.4 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        rows = {r["arm"]: r for r in comparisons
                if r["scenario"] == run["name"] and r["evaluation_budget"] == 8}
        reference = next(r for r in comparisons if r["scenario"] == run["name"]
                         and r["arm"] == "default" and r["evaluation_budget"] == 24)
        arms = run["summary"]["config"]["arms"]
        for ax, column, title in zip(
                axes[row], ("train_pooled_rmse_mm", "heldout_pooled_rmse_mm"),
                ("Training", "Held-out")):
            y = np.arange(len(arms))
            ax.barh(y, [rows[arm][column] for arm in arms],
                    color=[COLORS[arm] for arm in arms], alpha=.8)
            ax.axvline(reference[column], color="black", linestyle="--",
                       linewidth=1.5, label=f"default @24: {reference[column]:.2f} mm")
            ax.set_yticks(y, arms)
            ax.invert_yaxis()
            ax.set_xlabel("Pooled visible-point RMSE (mm)")
            ax.set_title(f"{run['name']} — {title} @8 saved checkpoint")
            ax.grid(axis="x", alpha=.25)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plot_files(fig, output_dir, "budget8_pooled_rmse")


def trajectory_figure(runs, all_frames, output_dir):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.4 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        ax = axes[row, 0]
        cutoff = run["summary"]["config"]["train_end"]
        for arm in run["summary"]["config"]["arms"]:
            budgets = sorted({r["evaluation_budget"] for r in all_frames
                              if r["scenario"] == run["name"] and r["arm"] == arm})
            if not budgets:
                continue
            for budget in ([0] if budgets[-1] == 0 else [0, budgets[-1]]):
                points = [r for r in all_frames if r["scenario"] == run["name"]
                          and r["arm"] == arm and r["evaluation_budget"] == budget]
                ax.plot([r["frame"] for r in points], [r["rmse_mm"] for r in points],
                        linestyle="--" if budget == 0 else "-",
                        alpha=.55 if budget == 0 else 1,
                        color=COLORS[arm], linewidth=1.5,
                        label=f"{arm} {'initial' if budget == 0 else f'best @ {budget}'}")
        ax.axvline(cutoff - .5, color="black", linestyle=":", linewidth=1,
                   label=f"train/test cutoff {cutoff}")
        ax.set_title(f"{run['name']} — per-frame visible-point RMSE")
        ax.set_xlabel("Frame")
        ax.set_ylabel("RMSE (mm)")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    plot_files(fig, output_dir, "trajectory_rmse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/nailoong_material_init/comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(run_dir) for run_dir in args.run_dir]
    comparisons = [row for run in runs for row in comparison_rows(run)]
    convergence = [row for run in runs for row in convergence_rows(run, comparisons)]
    frames = [row for run in runs for row in frame_rows(run)]
    write_csv(args.output_dir / "comparison.csv", comparisons)
    write_csv(args.output_dir / "convergence.csv", convergence)
    write_csv(args.output_dir / "frame_rmse.csv", frames)
    optimization_figure(runs, comparisons, args.output_dir)
    budget8_figure(runs, comparisons, args.output_dir)
    trajectory_figure(runs, frames, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
