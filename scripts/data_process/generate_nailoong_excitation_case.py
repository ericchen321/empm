"""Generate a separate same-solver Nailoong material-recovery diagnostic.

The forward MPM model and the later fit use the same solver. This tests recovery
under known excitation; it is not independent evidence for the VLM material prior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial import cKDTree

from phys_sim import InvPhyTrainerWarpMPM
from phys_sim.utils import cfg
from scripts.train_test.run_nailoong_material_init import configure_controllers, rollout
from scripts.utils import set_all_seeds


REPO = Path(__file__).resolve().parents[2]
FRAME_COUNT = 70
FRAME_DT = 1 / 30
SIM_DT_CEILING = 2e-4
TRUE_BODY_E_PA = 3000.0
TRUE_HARDWARE_E_PA = 30000.0
RING_PULL_M = 0.015
PRIOR_SCALE = 2.3648e-5


def ring_offsets() -> np.ndarray:
    # Frames 0:5 hold, 5:30 pull, 30:40 hold, 40:65 return, 65:70 hold.
    return np.concatenate((
        np.zeros(5), np.arange(1, 26) / 25,
        np.ones(10), np.arange(24, -1, -1) / 25, np.zeros(5)
    )) * RING_PULL_M


def generate(case_dir: Path, field_path: Path, output: Path) -> None:
    case_dir = case_dir.resolve()
    field_path = field_path.resolve()
    output = output.resolve()
    with (case_dir / "final_data.pkl").open("rb") as stream:
        source = pickle.load(stream)
    with np.load(field_path) as archive:
        source_field = {name: archive[name] for name in archive.files}
    labels = source_field["part_labels"]
    initial_object = source["object_points"][0]
    initial_controllers = source["controller_points"][0]
    controller_labels = source["controller_labels"]

    # Select the same semantic anchors as the fitting harness at frame 0.
    anchor_indices = {}
    for role, controller_label, part_label in (("ring", 0, 2), ("belly", 1, 0)):
        part_ids = np.flatnonzero(labels == part_label)
        distances, _ = cKDTree(initial_controllers[controller_labels == controller_label]).query(
            initial_object[part_ids]
        )
        anchor_indices[role] = int(part_ids[np.argmin(distances)])
    ring_anchor = initial_object[anchor_indices["ring"]]
    belly_anchor = initial_object[anchor_indices["belly"]]
    pull_axis = ring_anchor - belly_anchor
    pull_axis = pull_axis / np.linalg.norm(pull_axis)
    offsets_m = ring_offsets()[:, None] * pull_axis[None, :]

    controllers = np.repeat(initial_controllers[None], FRAME_COUNT, axis=0)
    controllers[:, controller_labels == 0] += offsets_m[:, None, :]
    seed_data = {
        "object_points": np.repeat(initial_object[None], FRAME_COUNT, axis=0),
        "object_colors": np.repeat(source["object_colors"][0:1], FRAME_COUNT, axis=0),
        "object_visibilities": np.ones((FRAME_COUNT, len(initial_object)), dtype=bool),
        "object_motions_valid": np.ones((FRAME_COUNT, len(initial_object)), dtype=bool),
        "controller_points": controllers,
        "controller_labels": controller_labels.copy(),
        "controller_mask": source["controller_mask"].copy(),
        "surface_points": source["surface_points"].copy(),
        "interior_points": source["interior_points"].copy(),
    }
    seed_data["object_motions_valid"][-1] = False
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "_generation_seed.pkl"
    with seed_path.open("wb") as stream:
        pickle.dump(seed_data, stream)

    substeps = math.ceil(FRAME_DT / SIM_DT_CEILING)
    sim_dt = FRAME_DT / substeps
    args = SimpleNamespace(
        start_frame=0, frame_dt=FRAME_DT, sim_dt=sim_dt,
        grid_size=32, grid_lim=.064, sim_shift=(.54, .47, .496),
        controller_halfsize=.003, controller_anchor="semantic", controller_vscale=None,
    )
    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = output.name
    cfg.train_frame = 40
    set_all_seeds(42)
    trainer = InvPhyTrainerWarpMPM(
        data_path=str(seed_path), base_dir=str(output), train_frame=40,
        save_artifacts=False,
        mpm_kwargs={"grid_size": args.grid_size, "grid_lim": args.grid_lim,
                    "frame_dt": FRAME_DT, "sim_dt": sim_dt},
        mpm_shift=args.sim_shift,
    )
    configure_controllers(trainer, labels, args)
    true_E_pa = np.where(labels == 0, TRUE_BODY_E_PA, TRUE_HARDWARE_E_PA).astype(np.float32)
    trajectory = rollout(trainer, true_E_pa, args, end_frame=FRAME_COUNT)
    object_points = trajectory.detach().cpu().numpy().astype(np.float32)
    seed_data["object_points"] = object_points
    with (output / "final_data.pkl").open("wb") as stream:
        pickle.dump(seed_data, stream)
    seed_path.unlink()

    # The fit loader requires exact agreement between the case's first frame and field.
    source_field["structure_points_m"] = object_points[0].copy()
    np.savez_compressed(output / "field.npz", **source_field)
    np.savez_compressed(output / "truth.npz", true_youngs_modulus_pa=true_E_pa,
                        part_labels=labels, structure_points_m=object_points[0])
    metadata = {
        "frame_num": FRAME_COUNT, "fps": 30, "data_kind": "same_solver_mpm_synthetic_recovery",
        "point_unit": "meter", "has_rgbd_images": False,
        "source_geometry_case": str(case_dir),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps(
        {"frame_len": FRAME_COUNT, "train": [0, 40], "test": [40, 70]}, indent=2
    ) + "\n")
    provenance = {
        "schema_version": "nailoong-same-solver-mpm-recovery-v1",
        "purpose": "Known-material synthetic identifiability and recovery diagnostic",
        "caveat": "Inverse crime: generation and fitting use the same EMPM solver. This is not independent VLM evidence and does not replace the original RGB-D case.",
        "source_geometry_case": str(case_dir),
        "source_material_field": str(field_path),
        "initial_object_point_count": int(len(initial_object)),
        "controller_label_roles": {"0": "ring", "1": "belly"},
        "anchor_object_indices": anchor_indices,
        "anchor_world_m": {"ring": ring_anchor.tolist(), "belly": belly_anchor.tolist()},
        "ring_pull_axis_world": pull_axis.tolist(),
        "ring_pull_max_m": RING_PULL_M,
        "motion_frames": {"initial_hold": [0, 5], "pull": [5, 30],
                          "extended_hold": [30, 40], "return": [40, 65],
                          "final_hold": [65, 70]},
        "motion_definition": "All controller-0 tracked points translate together along ring_pull_axis_world; controller-1 points remain fixed. Intervals are half-open frame ranges.",
        "generator_truth_pa": {"body": TRUE_BODY_E_PA, "hardware": TRUE_HARDWARE_E_PA},
        "generator_truth_file": "truth.npz",
        "fitting_initialization_not_used_for_generation": {
            "default_uniform_pa": 8000.0,
            "vlm_prior_scale": PRIOR_SCALE,
            "scaled_body_pa": float(5e6 * PRIOR_SCALE),
            "scaled_hardware_pa": float(2.2e9 * PRIOR_SCALE),
        },
        "common_mpm_config": {
            "material_type": "jelly", "poisson_ratio": .3, "density_kg_m3": 2000.0,
            "gravity_m_s2": -9.8, "grid_velocity_damping_scale": .98,
            "frame_dt_s": FRAME_DT, "sim_dt_s": sim_dt,
            "substeps_per_frame": substeps, "grid_size": 32, "grid_lim": .064,
            "sim_shift": list(args.sim_shift), "controller_halfsize_sim": .003,
            "controller_halfsize_world_m": .015,
        },
        "initialization_geometry_note": "field.npz structure_points_m is set to the generated frame-0 coordinates for exact fit-loader alignment.",
        "visibility": "All synthetic object tracks visible; adjacent motions valid except final frame.",
        "train_test_split": {"train": [0, 40], "test": [40, 70]},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated {output / 'final_data.pkl'} with {FRAME_COUNT} frames and "
          f"{len(initial_object)} object tracks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=REPO / "outputs/double_nailoong_grippers_20261011")
    parser.add_argument("--field", type=Path, default=REPO / "outputs/nailoong_material_init/field.npz")
    parser.add_argument("--output", type=Path,
                        default=REPO / "outputs/nailoong_mpm_synthetic_recovery")
    args = parser.parse_args()
    generate(args.case_dir, args.field, args.output)


if __name__ == "__main__":
    main()
