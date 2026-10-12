# Full synthetic Nailoong rollout with grippers

This extends the previous source-step-80–149 comparison to the complete saved
150-frame synthetic sequence, source steps 0–149 at 30 Hz. The video plays for
5 seconds. Saved states are after each control step, with timestamps 1/30–5 s;
MPM advances 149 transitions from the first saved state. The sequence includes
open grippers, grasping, a 45 mm lift, belly squeeze, and approximately 25 mm
ring pull. This reuses the full original action, rather than scaling its input
while incorrectly retaining the old ground truth.

The full-run metrics and videos below are **historical, pre-correction results**:
they use the original surface particles, a floor predicate that excluded the
exactly on-plane grid row, and −9.8 simulator units/s² gravity. A later
tetrahedral-volume replay with inclusive floor contact and correctly converted
world gravity is documented separately under “Collapse investigation and
corrected replay.” The historical results remain here to show the failure that
prompted that investigation; they are not results of the corrected replay.

## Fixed-parameter transfer

Both prediction arms reuse their saved eight-evaluation material fits:

| Fit initialization | Body E (Pa) | Hardware E (Pa) |
|---|---:|---:|
| Default | 13,189.77 | 13,189.77 |
| Semantic | 1,676.67 | 271,398.09 |

No material fitting is performed on this extended sequence. These are the
effective EMPM fields from the earlier experiment, not literal physical moduli.
The source FEM used the original VLM material asset. The full sequence overlaps
the fitting interval (source states 81–143), so it is a longer fixed-parameter
transfer diagnostic, not a wholly held-out sequence or independent validation
of the material prior.

`prepare_nailoong_full_sequence.py` transports the existing 1765 tracked-point
identities using their nearest source FEM triangles and local rigid transforms.
This preserves their source-step-80 positions exactly, along with the material
array ordering. Velocities use the instantaneous translation and angular
velocity fitted to the same state's saved FEM vertex velocities, without a
forward finite difference. Geometry correspondences and appearance still use
the source-step-80 reference, and the fields were fitted on states81–143.
Each of the 30 controller points is attached
to a recorded gripper link, preserving its source-step-80 position to numerical
precision. All quantities remain in the existing world coordinate system.

The historical solver used 334 substeps per frame, grid32 over 0.064 simulator
units, sqrt(0.98) grid damping per substep, and the sticky floor at world z=0.
It explicitly preserves the mapped initial positions instead of applying the
old near-floor 5 mm lift. Kinematic contact boxes activate for the source
39→40 belly transition and 40→41 ring transition. During each MPM rollout, later
object states are used only for evaluation/rendering; both predictions share
recorded gripper inputs. The MPM deformation gradient starts at identity:
initializing positions and velocities does not restore FEM prestress, including
in the source-state-49 diagnostic.
The contact model still represents moving grid-velocity boxes, not resolved
finger contact, squeeze pressure, or a volumetric FEM object.

A separate source-state-49 initialization diagnoses behavior after grasping,
before the lift. It starts a new, clearly labeled rollout and never pads the
full sequence with GT or resets either rollout during simulation. Its output
directory is `outputs/nailoong_full_sequence/rollout_from49/`.

## Historical pre-correction results and interpretation

RMSE is sqrt(mean(||predicted point - mapped FEM point||²)) in millimeters,
pooled over all 1765 points and every saved frame in the specified interval.
These source-mapped points include hidden points, so these numbers do not have
the same observation mask as the earlier RGB-D experiment. The initialized
first frame is included and has effectively zero error for both arms.

| Initialization state / metric | Default fit | Semantic fit |
|---|---:|---:|
| Source 0, entire 0–149 RMSE (mm) | 102.980 | 96.397 |
| Source 0, final-frame RMSE (mm) | 124.089 | 112.195 |
| Source 49, entire 49–149 RMSE (mm) | 23.240 | 14.696 |
| Source 49, final-frame RMSE (mm) | 40.243 | 17.221 |

In this original surface-particle replay, both full-sequence predictions
collapse toward the table before grasping;
their error is already approximately 80 mm by source step29. No particle leaves
the grid domain. This configuration does not preserve the initially standing
object. Subsequent controlled experiments found more than one contributing
configuration issue; this result alone does not identify surface sampling as
the sole cause. This historical full-run failure is retained in the archived
videos, without a GT prefix or reset.
The small difference between two large full-run errors does not establish
successful material transfer.

The separately initialized post-grasp rollout gives 36.8% lower pooled error
for the semantic fit. Its phase RMSEs (default/semantic, mm) are lift 13.93/11.47,
squeeze 20.01/15.38, and pull 31.50/16.54. The gain is concentrated in hardware:
ring 34.55/4.95, clasp 87.26/37.53, connector 23.53/1.30. Body error increases
from 7.92 to 12.93 mm. This is a conditional result with a known initial state,
not proof of all-part improvement or independent action generalization.

## Gaussian rendering

All three panels use the existing object appearance trained from source step80
(the previous clip's frame0). No appearance is fitted to the extended motion.
GT is **source-FEM-driven Gaussian rendering**, not observed RGB. The two
prediction panels are driven by the fixed-material MPM trajectories. This
shared appearance helps compare motion; it does not create an independent
image-space test, and initial occlusion holes and sparse-track skinning remain.

The six source gripper meshes are surface sampled into 173,386 anisotropic
Gaussians. The palm and two fingers of each gripper follow their separate
recorded rigid poses, including aperture changes. A further 25,972 Gaussians
represent the table top. These rigid surfaces come from known source geometry,
not an image-based gripper reconstruction. They are rendered together with the
33,724 object Gaussians, so depth sorting produces real mutual occlusion; no
observed silhouette or future depth masks clip the predictions.

The three calibrated camera poses provide separate GT/default/semantic streams and
three-column comparisons. Each has normal 30 fps playback and explicitly
labeled 3× slow playback at 10 fps, without interpolated frames. Camera1/2 focal
lengths are multiplied by 0.82 to widen the view and retain floor-level failed
predictions; the same framing is applied to all three arms. Camera0 is unchanged.

## Historical pre-correction videos

Each directory contains all four streams at both playback speeds. The complete sequence has 150 frames (5 s normal / 15 s slow); the post-grasp sequence has 101 frames (3.367 s normal / 10.1 s slow).

| Sequence | View | GT | Default | Semantic | Comparison | 3× slow comparison |
|---|---|---|---|---|---|---|
| Full 0–149 | Overview | [GT](../outputs/nailoong_full_sequence/videos/camera_0_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/videos/camera_0_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/videos/camera_0_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/videos/camera_0_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/videos/camera_0_comparison_slow_10fps.mp4) |
| Full 0–149 | Closeup | [GT](../outputs/nailoong_full_sequence/videos/camera_1_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/videos/camera_1_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/videos/camera_1_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/videos/camera_1_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/videos/camera_1_comparison_slow_10fps.mp4) |
| Full 0–149 | Oblique | [GT](../outputs/nailoong_full_sequence/videos/camera_2_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/videos/camera_2_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/videos/camera_2_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/videos/camera_2_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/videos/camera_2_comparison_slow_10fps.mp4) |
| Initialized at 49 | Overview | [GT](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_0_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_0_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_0_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_0_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_0_comparison_slow_10fps.mp4) |
| Initialized at 49 | Closeup | [GT](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_1_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_1_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_1_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_1_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_1_comparison_slow_10fps.mp4) |
| Initialized at 49 | Oblique | [GT](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_2_gt_full_30fps.mp4) | [Default](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_2_default_full_30fps.mp4) | [Semantic](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_2_semantic_full_30fps.mp4) | [Comparison](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_2_comparison_full_30fps.mp4) | [Slow](../outputs/nailoong_full_sequence/rollout_from49/videos/camera_2_comparison_slow_10fps.mp4) |

Original synthetic RGB references, with the complete source robot meshes and original camera framing: [overview](/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6/overview.mp4), [closeup](/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6/closeup.mp4), [oblique](/home/eric/research/HAG4R-Control/outputs/nailoong_dual_arm/run_v6/belly_oblique.mp4). These are distinct from the shared-appearance Gaussian GT panels.

## Collapse investigation and corrected replay

The source state-0 object is unsupported by grippers through state 39. A
controlled sequence kept the saved fitted material fields, initial positions
and velocities, controller schedule, grid, timestep, damping, and material law
fixed. The corrected representation uses 24,114 positive-volume particles at
the source tetrahedron centroids, with each particle's source tetrahedron
volume. The original 1,765 tracked points remain as **zero-volume observation
markers** for the same RMSE and Gaussian rendering identities. The source
tetrahedra supply their part labels; the fitted effective E fields are retained
without refitting or physical-modulus reinterpretation.

| Source 0–39 replay | Changed from preceding run | Default passing states / 40 | Semantic passing states / 40 | Material-body rotation at 39, default / semantic |
|---|---|---:|---:|---:|
| H0 tet volume | Solid material quadrature in place of surface-only material | 16 | 2 | 79.74° / 74.20° |
| H1 tet volume | Floor includes grid nodes exactly on its plane (`<= 0`) | 32 | 2 | 27.49° / 75.02° |
| H2 tet volume | World −9.81 m/s² maps to simulator −1.962 units/s² | **40** | **40** | **0.97° / 5.25°** |

“Semantic” here is the saved `spatial_eval8` fit. Passing a state requires
the observed body's z95−z05 span to be at least 80% of the same-state source
span and its median z to be within 15 mm of the source median. Both limits must
hold at every state. The source state-39 span/median is 91.783/64.771 mm.
H2's state-39 observed span/median is 91.651/63.678 mm for default and
88.657/58.489 mm for semantic; its worst span ratios are 0.997/0.966 and
worst median deviations are 1.10/6.28 mm. H2's pooled source 0–39 RMSE is
2.024/5.928 mm, respectively. The material rotation measurements show that
the simulated solid itself remains upright in this short replay, alongside
the observation markers.

The H0 solid experiment shows that adding interior material alone did not
restore standing. The isolated H1 floor correction substantially reduced
default tipping, but semantic still tipped. Coordinates use
`x_sim = x_world / 5 + shift`, and time remains in seconds: the earlier
simulator gravity −9.8 therefore corresponded to world −49 m/s².
H2 uses physical world −9.81 m/s² and the corresponding simulator −1.962.
These interventions support a combined explanation for the unsupported-phase
failure in this setup; they do not establish a single sole cause. The original
fits remain effective EMPM parameters, so this comparison does not establish a
new material-fitting benefit.

The [H0 validation](../.agents/reports/nailoong_collapse_h0_validation.md)
and [H1 validation](../.agents/reports/nailoong_collapse_h1_validation.md)
record the isolated comparisons. H2's [summary](../outputs/nailoong_collapse/h2_tet_volume_0_39/summary.json)
and [frame metrics](../outputs/nailoong_collapse/h2_tet_volume_0_39/frame_metrics.csv)
record the actual two-fit short rollout.

The corrected **continuous source 0–149 replay** starts at source state 0,
advances all 149 transitions without reset, and uses the same fixed fitted
fields. Its [summary](../outputs/nailoong_collapse/h2_tet_volume_0_149/summary.json),
[frame metrics](../outputs/nailoong_collapse/h2_tet_volume_0_149/frame_metrics.csv),
and [phase metrics](../outputs/nailoong_collapse/h2_tet_volume_0_149/metrics.csv)
are separate from the archived surface results above:

| Corrected full replay | Default fit | Semantic fit |
|---|---:|---:|
| Pooled source 0–149 RMSE (mm) | 3.607 | 9.072 |
| Final-frame RMSE (mm) | 5.145 | 9.890 |
| Observed-body median rise, state 49→79 (mm) | 43.908 | 43.325 |
| Material-body median rise, state 49→79 (mm) | 42.985 | 42.729 |

The source observed-body median rises about 44.4 mm over the same interval.
The camera-1 comparison shows both corrected solids upright before grasp and
coherent through the lift and later motion, without renewed collapse. Watch the
[normal-speed comparison](../outputs/nailoong_collapse/h2_tet_volume_0_149/videos/camera_1_comparison_full_30fps.mp4)
(150 frames, 5 s) or the
[3× slow comparison](../outputs/nailoong_collapse/h2_tet_volume_0_149/videos/camera_1_comparison_slow_10fps.mp4)
(150 frames, 15 s). These are source-FEM-driven Gaussian GT and two MPM panels
with shared appearance, as described above. The corrected fixed-parameter
replay favors the default fit overall; it is not evidence of a new semantic
prior benefit. The earlier fitting comparison was made under the old model
configuration and needs reevaluation before transferring its material-prior
conclusion to this corrected model. No new optimization was performed here.

## Reproduce and provenance

Run from `/home/eric/research/empm` with the existing `.conda/empm` environment:

The current corrected replay uses the already prepared source tracks and
saved eight-evaluation fits. These commands create fresh, separate outputs;
the second simulation continues from source state 0 through 149 without a
source-state-49 reset.

```bash
.conda/empm/bin/python scripts/data_process/prepare_nailoong_full_sequence.py
.conda/empm/bin/python scripts/eval/rollout_nailoong_full_sequence.py \
  --representation tet_volume --end-source-step 39 \
  --output-dir outputs/nailoong_collapse/h2_tet_volume_0_39
.conda/empm/bin/python scripts/eval/rollout_nailoong_full_sequence.py \
  --representation tet_volume \
  --output-dir outputs/nailoong_collapse/h2_tet_volume_0_149

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.conda/empm/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.conda/empm/var/torch_extensions"
export MAX_JOBS=8
.conda/empm/bin/python scripts/gs/render_nailoong_full_sequence.py \
  --predictions outputs/nailoong_collapse/h2_tet_volume_0_149/trajectories.npz \
  --output-dir outputs/nailoong_collapse/h2_tet_volume_0_149/videos \
  --cameras 1
```

The commands below generated the **historical surface-replay artifacts** when
the older solver/contact/gravity configuration was in use. They document
provenance; rerunning them against the current code uses the corrected floor
and gravity and will not recreate the archived numerical results.

```bash
.conda/empm/bin/python scripts/data_process/prepare_nailoong_full_sequence.py
.conda/empm/bin/python scripts/eval/rollout_nailoong_full_sequence.py
.conda/empm/bin/python scripts/eval/rollout_nailoong_full_sequence.py \
  --start-source-step 49 --output-dir outputs/nailoong_full_sequence/rollout_from49

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.conda/empm/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$PWD/.conda/empm/var/torch_extensions"
export MAX_JOBS=8
.conda/empm/bin/python scripts/gs/render_nailoong_full_sequence.py
.conda/empm/bin/python scripts/gs/render_nailoong_full_sequence.py \
  --predictions outputs/nailoong_full_sequence/rollout_from49/trajectories.npz \
  --output-dir outputs/nailoong_full_sequence/rollout_from49/videos
```

Mapping provenance is recorded in `source_mapping.json`; each rollout saves
`trajectories.npz`, `summary.json`, `metrics.csv`, and `frame_metrics.csv`.
Rendering records stream filenames, frame counts and playback rates in
`video_inventory.json` in each video directory.
