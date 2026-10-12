"""Recover pale eye regions omitted by the synthetic color-only object mask."""

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt


def load_object_mask(case_dir, camera, frame):
    case_dir = Path(case_dir)
    mask_dir = case_dir / "mask" / str(camera)
    mask = cv2.imread(str(mask_dir / "0" / f"{frame}.png"), 0) > 0
    grippers = ((cv2.imread(str(mask_dir / "1" / f"{frame}.png"), 0) > 0)
                | (cv2.imread(str(mask_dir / "2" / f"{frame}.png"), 0) > 0))
    depth = np.load(case_dir / "depth" / str(camera) / f"{frame}.npy") / 1000
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8)).astype(bool)
    holes = binary_fill_holes(closed) & ~mask
    distance, nearest = distance_transform_edt(~mask, return_indices=True)
    neighbor_depth = depth[tuple(nearest)]
    # True ring openings see the distant scene and are not filled. Restrict
    # additions to enclosed image holes at the existing object's depth.
    fill = (holes & ~grippers & (distance <= 20) & (depth > .2)
            & (np.abs(depth - neighbor_depth) < .012))
    return mask | fill
