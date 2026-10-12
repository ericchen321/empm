"""Calibrated HAG4R cameras for world-coordinate Nailoong Gaussian renders."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from third_party.gaussian_splatting.scene.cameras import MiniCam
from third_party.gaussian_splatting.utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
)


def load_case_cameras(case_dir, camera_indices=(1, 2)):
    """Return ``{camera_index: MiniCam}`` in the case's world coordinates.

    The case's RGB-D lifting uses x=(u+metadata_cx-width)*z/fx, so its
    effective projection center is width-metadata_cx.
    """
    case_dir = Path(case_dir)
    metadata = json.loads((case_dir / "metadata.json").read_text())
    with (case_dir / "calibrate.pkl").open("rb") as stream:
        c2ws = np.asarray(pickle.load(stream))
    width, height = metadata["WH"]
    cameras = {}
    for index in camera_indices:
        k = np.asarray(metadata["intrinsics"][index], dtype=np.float32).copy()
        k[0, 2] = width - k[0, 2]
        fovx = focal2fov(float(k[0, 0]), width)
        fovy = focal2fov(float(k[1, 1]), height)
        w2c = np.linalg.inv(c2ws[index]).astype(np.float32)
        world_view = torch.from_numpy(w2c).transpose(0, 1).cuda()
        projection = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
        camera = MiniCam(
            width, height, fovy, fovx, 0.01, 100.0,
            world_view, world_view @ projection,
        )
        camera.K = k
        cameras[index] = camera
    return cameras
