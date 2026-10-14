# Corrected Nailoong semantic-material refit

The fixed-budget H4 experiment **narrowly achieved** its stated synthetic-sequence goal: the semantically initialized fit has lower pooled full-sequence point RMSE than an equally capable neutral fit, and its fitted body is softer than its hardware. The full-sequence gain is about 0.7%; the temporal holdout and final frame are worse, and the four parts do not all improve. This is a result for the recorded source sequence and model, not evidence of general test-time superiority. The [formal H4 verdict](../.agents/reports/nailoong_semantic_refit_h4_verdict.md) confirms the causal readout repair and this qualified result; no H5 was needed for the stated goal.

## Shared model and comparison

Both arms start from source state 0 and use the same continuous EMPM rollout, source velocities, original 1,765 point identities and visibility mask, recorded rigid-gripper poses, tet-volume material discretization, 32-cell grid, 334 substeps per 30 Hz frame, jelly constitutive law, ν=0.3, density parameter 2,000, frictional floor (coefficient 0.9), and recorded-finger grid-centroid controller. H4 predicts each observation from its fixed source-0 material-face association and its owner's simulated center and endpoint deformation; the readout does not feed back into dynamics. Raw grid markers remain a diagnostic. The [H4 validation](../.agents/reports/nailoong_semantic_h4_validation.md) records the causal same-physics readout checks.

Both fits optimize **two free log Young's moduli**, body and shared hardware, with the same seed 42, Nelder–Mead simplex log offsets `(0,0), (0.5,0), (0,0.5)`, 48 forward evaluations per arm, and `xatol=fatol=1e-10`. The neutral two-E baseline starts at **8,000/8,000 Pa**. The VLM-derived semantic prior supplies a **440:1 hardware/body ratio**, normalized by source-0 tet volume to the same 8,000 Pa arithmetic mean: **365.4465/160,796.4675 Pa**. There is no stiffness-order constraint or prior penalty. The objective is pooled visible squared Euclidean world-mm error on source states **1–134**; the lowest finite training loss, including the seed, selects each field. This is forward numerical optimization, not native adjoint training. States **135–149** do not select parameters but were inspected during model development, so they are a temporal holdout rather than untouched validation. Full RMSE covers states **0–149**, mostly fitted frames.

| Arm | Selected call / 48 | Body / hardware E (Pa) | Hardware/body | Fit train MSE (mm²) |
|---|---:|---:|---:|---:|
| Neutral default | 40 | 209,810.202 / 105,784.513 | 0.504 | 1.927378 |
| Semantic initialization | 46 | 177,891.466 / 485,198.022 | 2.727 | 1.863164 |

Each arm used all 48 calls; four neutral and eight semantic calls failed numerically and still consumed budget. Neither search met the tight convergence tolerances. The selected E values are effective parameters of this simulator, not recovered physical source moduli.

## Independent full-sequence replays

RMSE below is `sqrt(mean(||prediction-target||²))` over the original visible point-frames in the stated interval, in mm. Each replay uses both training-selected fields without reselection. The primary was designated before inspecting the repeat.

| Replay | Arm | Train 1–134 | Full 0–149 | Holdout 135–149 | Final 149 |
|---|---|---:|---:|---:|---:|
| Primary | Neutral | 1.388834 | 1.589170 | 2.832575 | 2.944312 |
| Primary | Semantic | **1.365060** | **1.578249** | 2.874440 | 2.995831 |
| Repeat | Neutral | 1.388621 | 1.588973 | 2.832397 | 2.944119 |
| Repeat | Semantic | **1.365728** | **1.578731** | 2.874259 | 2.995277 |

The semantic full-RMSE lead is **0.010922 mm (0.687%)** in the primary and **0.010241 mm (0.645%)** in the repeat. Its holdout RMSE is about **0.04186 mm higher** in both. Primary full per-part RMSE improves for body **1.612163→1.597851** and clasp **1.691991→1.620820**, but worsens for connector **1.306385→1.324295** and ring **1.320759→1.409336**; the repeat has the same direction. Primary mean point error improves while median point error worsens (1.190395→1.209408 mm). The source FEM was generated using the material asset that supplied the semantic prior. Thus this is prior-matched synthetic evidence, with a modest, mixed training-dominated benefit, not independent evidence that semantic initialization generally improves held-out behavior.

Source results: [fit summary/history](../outputs/nailoong_semantic_refit/h4_fit_retry1_20261013/summary.json), [primary replay](../outputs/nailoong_semantic_refit/h4_replay_20261013/summary.json), [repeat replay](../outputs/nailoong_semantic_refit/h4_replay_repeat_20261013/summary.json), and [detailed replay audit](../outputs/nailoong_semantic_refit/h4_fit_replay_analysis.json). The [primary figure](../outputs/nailoong_semantic_refit/h4_final_primary.png) ([PDF](../outputs/nailoong_semantic_refit/h4_final_primary.pdf), [plotted metrics](../outputs/nailoong_semantic_refit/h4_final_primary.metrics.json)) shows all 48 calls, frame/part errors and selected E. The [camera-0 GT/default/semantic Gaussian comparison](../outputs/nailoong_semantic_refit/h4_gaussian_primary_20261013/videos/camera_0_comparison_full_30fps.mp4) renders the **primary** material-face trajectories at 30 fps for 150 frames; [provenance](../outputs/nailoong_semantic_refit/h4_gaussian_primary_20261013/render_provenance.json) identifies the output-local key aliases and recorded rigid-gripper assets. It is a qualitative Gaussian visualization, not an additional error metric.

## Reproduce in fresh output directories

Run from the repository root with the existing source tracks, case, source FEM run, material field and `.conda/empm` environment. The commands leave the recorded inputs and completed result directories intact:

```bash
.conda/empm/bin/python -u scripts/train_test/run_nailoong_corrected_refit.py --output-dir outputs/nailoong_semantic_refit/h4_fit_fresh --controller-mode recorded_finger_grid_centroids --observation-readout material_faces --observation-field outputs/nailoong_material_init/field.npz
.conda/empm/bin/python -u scripts/eval/rollout_nailoong_full_sequence.py --run-dir outputs/nailoong_semantic_refit/h4_fit_fresh --output-dir outputs/nailoong_semantic_refit/h4_replay_fresh --representation tet_volume --evaluations 48 --arms default spatial --start-source-step 0 --end-source-step 149
.conda/empm/bin/python -u scripts/eval/rollout_nailoong_full_sequence.py --run-dir outputs/nailoong_semantic_refit/h4_fit_fresh --output-dir outputs/nailoong_semantic_refit/h4_replay_repeat_fresh --representation tet_volume --evaluations 48 --arms default spatial --start-source-step 0 --end-source-step 149
.conda/empm/bin/python outputs/nailoong_semantic_refit/plot_h4_final_results.py --fit-dir outputs/nailoong_semantic_refit/h4_fit_fresh --replay-dir outputs/nailoong_semantic_refit/h4_replay_fresh --output-prefix outputs/nailoong_semantic_refit/h4_final_fresh_primary
```

For a fresh primary video, create the renderer's historical-key alias archive from the **fresh** `eval48` material-face trajectory and use the output-local wrapper, whose visible labels explicitly say `fitted @48 | material-face readout`:

```bash
mkdir -p outputs/nailoong_semantic_refit/h4_gaussian_fresh
.conda/empm/bin/python - <<'PY'
import numpy as np
from pathlib import Path
a = np.load('outputs/nailoong_semantic_refit/h4_replay_fresh/trajectories.npz')
p = Path('outputs/nailoong_semantic_refit/h4_gaussian_fresh/renderer_aliases.npz')
np.savez_compressed(p, gt_world_m=a['gt_world_m'], default_eval8_world_m=a['default_eval48_world_m'], spatial_eval8_world_m=a['spatial_eval48_world_m'], source_steps=a['source_steps'], time_s=a['time_s'])
PY
CUDA_HOME=/usr/local/cuda-12.8 PATH="$PWD/.conda/empm/bin:/usr/local/cuda-12.8/bin:$PATH" TORCH_EXTENSIONS_DIR="$PWD/.conda/empm/var/torch_extensions" .conda/empm/bin/python outputs/nailoong_semantic_refit/h4_gaussian_primary_20261013/render_primary.py --predictions outputs/nailoong_semantic_refit/h4_gaussian_fresh/renderer_aliases.npz --output-dir outputs/nailoong_semantic_refit/h4_gaussian_fresh/videos --cameras 0
```
