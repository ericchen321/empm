"""Fixed source0 material-face association and read-only observation reconstruction."""

from collections import Counter
from pathlib import Path

import numpy as np

from scripts.data_process.prepare_nailoong_full_sequence import rigid_face_transfer


def prepare_material_faces(source_run, field_path, source0_observations_world_m,
                           observation_labels):
    """Associate saved source faces with unique exterior tets using topology only."""
    source_run = Path(source_run)
    with np.load(field_path) as field:
        face_ids = field["nearest_source_face_vertex_ids"].astype(np.int64)
        field_labels = field["part_labels"].astype(np.int64)
    with np.load(source_run / "initial_state.npz") as initial:
        tets = initial["tetrahedra"].astype(np.int64)
        tet_labels = initial["tet_part_labels"].astype(np.int64)
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    incidence = Counter(tuple(sorted(tet[list(corners)]))
                        for tet in tets for corners in faces)
    exterior_owner = {
        tuple(sorted(tet[list(corners)])): tet_id
        for tet_id, tet in enumerate(tets) for corners in faces
        if incidence[tuple(sorted(tet[list(corners)]))] == 1
    }
    owner_ids = np.array([exterior_owner[tuple(sorted(face))] for face in face_ids],
                         dtype=np.int64)
    assert np.array_equal(field_labels, observation_labels)
    assert np.array_equal(tet_labels[owner_ids], observation_labels)
    unique_ids, owner_index = np.unique(owner_ids, return_inverse=True)
    with np.load(source_run / "states" / "000000.npz") as source0:
        rest_vertices = source0["positions_m"][face_ids].astype(np.float64)
    return {
        "material_face_vertex_ids": face_ids,
        "material_face_owner_tet_ids": owner_ids,
        "material_face_unique_owner_tet_ids": unique_ids,
        "material_face_unique_owner_full_ids": unique_ids + len(face_ids),
        "material_face_owner_index": owner_index,
        "material_face_rest_vertices_world_m": rest_vertices,
        "material_face_rest_observations_world_m": np.asarray(
            source0_observations_world_m, dtype=np.float64).copy(),
    }


def _failure(prepared, frame, component, reason, owner_indices=None,
             observation_ids=None):
    inverse = prepared["material_face_owner_index"]
    if observation_ids is None:
        observation_ids = np.flatnonzero(np.isin(inverse, owner_indices))
    observation_ids = np.asarray(observation_ids, dtype=np.int64)
    if owner_indices is None:
        owner_indices = np.unique(inverse[observation_ids])
    owner_ids = prepared["material_face_unique_owner_tet_ids"][owner_indices]
    return {
        "first_failed_frame": int(frame),
        "component": component,
        "reason": reason,
        "affected_owner_tet_ids": owner_ids.tolist(),
        "affected_observation_ids": observation_ids.tolist(),
    }


def reconstruct_material_faces(prepared, owner_initial_world_m, owner_world_m,
                               owner_F_trial):
    """Use owner F on face corners, then the source triangle's proper rigid transfer."""
    X = np.asarray(owner_initial_world_m, dtype=np.float64)
    x = np.asarray(owner_world_m, dtype=np.float64)
    F = np.asarray(owner_F_trial, dtype=np.float64)
    inverse = prepared["material_face_owner_index"]
    Q = prepared["material_face_rest_vertices_world_m"].astype(np.float64)
    y0 = prepared["material_face_rest_observations_world_m"].astype(np.float64)
    offset = Q - X[inverse, None, :]
    predictions = []
    for frame in range(len(x)):
        bad_x = np.flatnonzero(~np.isfinite(x[frame]).all(axis=1))
        if len(bad_x):
            return _failure(prepared, frame, "owner_x", "nonfinite", bad_x)
        bad_F = np.flatnonzero(~np.isfinite(F[frame]).all(axis=(1, 2)))
        if len(bad_F):
            return _failure(prepared, frame, "owner_F_trial", "nonfinite", bad_F)
        q = x[frame, inverse, None, :] + np.einsum(
            "nij,nkj->nki", F[frame, inverse], offset)
        bad_q = np.flatnonzero(~np.isfinite(q).all(axis=(1, 2)))
        if len(bad_q):
            return _failure(prepared, frame, "face_corners", "nonfinite", observation_ids=bad_q)
        try:
            y = rigid_face_transfer(Q, y0, q)
        except np.linalg.LinAlgError as error:
            return _failure(prepared, frame, "rigid_face_transfer",
                            f"SVD failure: {error}",
                            observation_ids=np.arange(len(inverse)))
        bad_y = np.flatnonzero(~np.isfinite(y).all(axis=1))
        if len(bad_y):
            return _failure(prepared, frame, "rigid_face_transfer", "nonfinite_output",
                            observation_ids=bad_y)
        prediction = y.astype(np.float32)
        bad_prediction = np.flatnonzero(~np.isfinite(prediction).all(axis=1))
        if len(bad_prediction):
            return _failure(prepared, frame, "rigid_face_transfer",
                            "nonfinite_float32_output", observation_ids=bad_prediction)
        predictions.append(prediction)
    return np.stack(predictions)
