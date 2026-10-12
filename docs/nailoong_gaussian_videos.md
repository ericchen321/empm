# Nailoong Gaussian-rendered material-fit videos

The videos compare observed RGB with Gaussian renders driven by the saved
default and semantic material fits at the same **eight-evaluation budget**.
They use the existing 70-frame case and calibrated overview camera 0,
closeup camera 1, and belly-oblique camera 2. The simulations are not refitted
for these videos.

## Shared appearance and motion

`scripts/gs/train_nailoong_gaussians.py` trains one shared, 33,724-Gaussian
appearance from **frame 0 only**, using all three initial RGB-D views. Gaussian
means stay at the reconstructed world-coordinate surface points. Color,
opacity, scale, and rotation are fitted for 1800 iterations. Color is bounded
to [0,1], and a soft support-size prior prevents unobserved contact regions
from developing long splats. No later object image trains this appearance.

The original color-only segmentation omitted pale eye pixels. The shared
`nailoong_object_mask.py` fills enclosed mask holes only when their depth
agrees with adjacent object pixels, excluding gripper pixels. Actual ring
openings remain empty because the scene behind them has a different depth.

`scripts/gs/render_nailoong_comparison.py` uses a fixed mapping from each
Gaussian to four frame-0 tracked points of the same semantic part. Local
rotations are estimated from 16 neighbors in that part; rotations and
translations drive linear blend skinning. Both arms use exactly the same
appearance, mapping, camera, and background. Their trajectories are
`default_eval8_world_m` and `spatial_eval8_world_m` from
`outputs/nailoong_material_init/experiment_refined/predictions.npz`.

GT is **observed RGB under the object mask**, not a Gaussian model fitted to
future frames. These are object-only views on a common light-grey background.
Observed nonobject depth supplies the common gripper/scene occlusion layer
where it is at least 1 mm closer than the rendered object. Predicted silhouettes
are not clipped to the GT object silhouette.

## Video files

All files are under `outputs/nailoong_gaussian_videos/comparison/`.
For each camera (`0`, `1`, or `2`) there are four streams: `gt`, `default`,
`semantic`, and `comparison` (three panels side by side).

- `camera_<id>_<stream>_full_30fps.mp4`: 70 frames at 30 fps, 2.333 s playback.
- `camera_<id>_<stream>_heldout_3fps.mp4`: frames 64–69 at 3 fps, 2 s playback,
  visibly labeled **10x slow playback**. No interpolated frames are added.

The full videos mark frame 0 as INITIAL, frames 1–63 as TRAIN, and frames
64–69 as HELD OUT. Time labels use the original 30 Hz clock; the last frame
is at 2.30 s. Independent streams are 640×548, and side-by-side streams
are 1920×548. All are H.264/yuv420p MP4 with fast-start metadata.

| View | GT | Default | Semantic | Side by side |
|---|---|---|---|---|
| Overview, full | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_0_gt_full_30fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_0_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_0_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_0_comparison_full_30fps.mp4) |
| Overview, hold-out slow | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_0_gt_heldout_3fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_0_default_heldout_3fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_0_semantic_heldout_3fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_0_comparison_heldout_3fps.mp4) |
| Closeup, full | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_1_gt_full_30fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_1_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_1_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_1_comparison_full_30fps.mp4) |
| Closeup, hold-out slow | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_1_gt_heldout_3fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_1_default_heldout_3fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_1_semantic_heldout_3fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_1_comparison_heldout_3fps.mp4) |
| Oblique, full | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_2_gt_full_30fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_2_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_2_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_2_comparison_full_30fps.mp4) |
| Oblique, hold-out slow | [GT](../outputs/nailoong_gaussian_videos/comparison/camera_2_gt_heldout_3fps.mp4) | [Default](../outputs/nailoong_gaussian_videos/comparison/camera_2_default_heldout_3fps.mp4) | [Semantic](../outputs/nailoong_gaussian_videos/comparison/camera_2_semantic_heldout_3fps.mp4) | [Comparison](../outputs/nailoong_gaussian_videos/comparison/camera_2_comparison_heldout_3fps.mp4) |

## Interpretation

Missing surfaces behind the initial grippers are not reconstructed from
future frames. Sparse tracked-point skinning also contributes blur and
fragmentation; an observed-track-driven Gaussian diagnostic is saved for
frame 69 to expose this approximation. Large ring/clasp motions and body
distortions in the predictions are retained from the fitted MPM trajectories.
These are qualitative visualizations of that model, not independent
photometric generalization measurements. See the material-initialization
report for the positional errors and their limitations.

## Reproduce

Run from the repository root with `.conda/empm` and the system CUDA toolkit:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.conda/empm/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.conda/empm/var/torch_extensions"
export MAX_JOBS=8
.conda/empm/bin/python scripts/gs/train_nailoong_gaussians.py --iterations 1800
.conda/empm/bin/python scripts/gs/render_nailoong_comparison.py --cameras 0 1 2
```

`appearance/point_cloud.ply`, `appearance/metadata.json`, initial GT/render
pairs, per-frame PNGs, and training history accompany the videos under
`outputs/nailoong_gaussian_videos/`.

All 24 exported MP4 files were inspected with ffprobe: full videos have
70 frames at 30 fps; hold-out videos have six frames at 3 fps. Representative
initial and final rendered frames and decoded hold-out MP4 frames were
visually inspected. `comparison/video_inventory.json` records the actual
encoded formats and durations.
