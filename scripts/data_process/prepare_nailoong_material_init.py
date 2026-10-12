"""Map Nailoong's existing semantic FEM materials to EMPM's tracked points.

This prepares a material initialization artifact; it does not change training data
or fit material properties. The source simulation already uses these properties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import cKDTree
import yaml


REPO = Path(__file__).resolve().parents[2]


def prepare(case_dir: Path, hag4r_root: Path, output: Path) -> None:
    case_dir = case_dir.resolve()
    hag4r_root = hag4r_root.resolve()
    output = output.resolve()
    metadata = json.loads((case_dir / "metadata.json").read_text())
    run = hag4r_root / metadata["source_run"]
    source_step = metadata["source_steps"][0]
    asset = hag4r_root / "assets/objects/nailoong"
    run_metadata = json.loads((run / "run.json").read_text())
    run_config = yaml.safe_load((run / "config.yaml").read_text())

    with (case_dir / "final_data.pkl").open("rb") as stream:
        data = pickle.load(stream)
    structure_points = np.concatenate(
        (data["object_points"][0], data["surface_points"], data["interior_points"]), axis=0
    )

    with np.load(run / "initial_state.npz") as initial:
        tetrahedra = initial["tetrahedra"]
        tet_part_labels = initial["tet_part_labels"]
    with np.load(run / "states" / f"{source_step:06d}.npz") as state:
        vertices = state["positions_m"]
    with np.load(asset / "heterogeneous_params.npz") as materials:
        part_indices = materials["part_indices"]
        part_E_nu = materials["part_E_nu"]
        part_density = materials["part_density"]
    inferred = json.loads((asset / "inferred_params.json").read_text())
    names = {row["part_index"]: row["part_name"] for row in inferred["predictions"]}

    # Each external triangular face has one tetrahedral owner and its part label.
    faces = np.concatenate(
        [tetrahedra[:, ids] for ids in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    )
    owner = np.tile(np.arange(len(tetrahedra)), 4)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = counts[inverse] == 1
    boundary_faces = faces[boundary]
    boundary_labels = tet_part_labels[owner[boundary]]
    face_centers = vertices[boundary_faces].mean(axis=1)
    distance, nearest_face = cKDTree(face_centers).query(structure_points)
    labels = boundary_labels[nearest_face]

    E_by_label = dict(zip(part_indices.tolist(), part_E_nu[:, 0].tolist()))
    nu_by_label = dict(zip(part_indices.tolist(), part_E_nu[:, 1].tolist()))
    rho_by_label = dict(zip(part_indices.tolist(), part_density.tolist()))
    youngs = np.array([E_by_label[int(label)] for label in labels], dtype=np.float64)
    poisson = np.array([nu_by_label[int(label)] for label in labels], dtype=np.float64)
    density = np.array([rho_by_label[int(label)] for label in labels], dtype=np.float64)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "field.npz",
        structure_points_m=structure_points,
        part_labels=labels,
        youngs_modulus_pa=youngs,
        poisson_ratio=poisson,
        density_kg_m3=density,
        nearest_source_face_index=nearest_face,
        nearest_source_face_vertex_ids=boundary_faces[nearest_face],
        nearest_source_face_centroid_m=face_centers[nearest_face],
        nearest_distance_m=distance,
    )
    part_rows = []
    for part_index in part_indices:
        selected = labels == part_index
        part_rows.append({
            "part_index": int(part_index),
            "part_name": names[int(part_index)],
            "point_count": int(selected.sum()),
            "youngs_modulus_pa": E_by_label[int(part_index)],
            "poisson_ratio": nu_by_label[int(part_index)],
            "density_kg_m3": rho_by_label[int(part_index)],
            "nearest_face_distance_median_m": float(np.median(distance[selected])),
            "nearest_face_distance_p95_m": float(np.quantile(distance[selected], 0.95)),
        })
    provenance = {
        "schema_version": "nailoong-empm-material-initialization-v1",
        "field_path": "field.npz",
        "case_dir": str(case_dir),
        "source_run": str(run),
        "source_step": source_step,
        "source_state_convention": "Saved state after control step; frame 0 corresponds to source_step.",
        "material_source": str(asset / "heterogeneous_params.npz"),
        "semantic_source": str(asset / "inferred_params.json"),
        "generator": {
            "run_metadata_path": str(run / "run.json"),
            "run_config_path": str(run / "config.yaml"),
            "asset_name": run_config["scene"]["object_asset"],
            "material_override": run_config["scene"].get("object_material_params_file"),
            "control_frequency_hz": run_config["control"]["control_freq"],
            "simulation_substeps_per_control_step": run_config["control"]["simulation_substeps_per_control_step"],
            "source_control_steps_in_case": metadata["source_steps"],
            "phases": run_metadata["phases"],
            "contact_mechanism": run_metadata["contact_mechanism"],
            "run_resume": run_metadata.get("resume"),
        },
        "camera": {
            "original_resolution_wh": run_config["camera_resolution"],
            "exported_resolution_wh": metadata["WH"],
            "exported_fps": metadata["fps"],
            "exported_camera_names": metadata["camera_names"],
            "original_camera_specs": run_config["cameras"],
            "exported_intrinsics": metadata["intrinsics"],
            "exported_static_camera_to_world_path": str(case_dir / "calibrate.pkl"),
            "rgbd_exporter_source": str(hag4r_root / "scripts/export_nailoong_empm.py"),
            "depth_unit": metadata["depth_unit"],
        },
        "mapping": "Nearest boundary FEM triangle centroid in world coordinates at source_step; triangle inherits its sole tetrahedron owner's part label.",
        "structure_point_order": "final_data.pkl object_points[0], then surface_points, then interior_points, matching RealData.structure_points",
        "source_material_status": "The Genesis source rollout used this heterogeneous material file. These values are both a VLM-derived prior and the synthetic generator values, not independent experimental ground truth.",
        "mapping_limitations": [
            "Nearest boundary-face centroid is an approximate semantic transfer, especially near part interfaces.",
            "RGB-D tracks cover visible surfaces only; the far positive-y belly grasp surface is not represented at frame 0.",
            "The processed Nailoong mask captures only part of the top ring.",
        ],
        "object_track_count": int(len(data["object_points"][0])),
        "structure_point_count": int(len(structure_points)),
        "nearest_face_distance_median_m": float(np.median(distance)),
        "nearest_face_distance_p95_m": float(np.quantile(distance, 0.95)),
        "nearest_face_distance_max_m": float(distance.max()),
        "parts": part_rows,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"field": str(output / "field.npz"), "parts": part_rows,
                      "max_distance_m": provenance["nearest_face_distance_max_m"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=REPO / "outputs/double_nailoong_grippers_20261011")
    parser.add_argument("--hag4r-root", type=Path,
                        default=Path("/home/eric/research/HAG4R-Control"))
    parser.add_argument("--output", type=Path, default=REPO / "outputs/nailoong_material_init")
    args = parser.parse_args()
    prepare(args.case_dir, args.hag4r_root, args.output)


if __name__ == "__main__":
    main()
