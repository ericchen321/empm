# Nailoong material-initialization pilot

Date: 2026-10-12. Code and commands use the repository's `.conda/empm` environment.

Status: gradient repair, both four-arm studies, and the fixed replay protocol are complete. The refined study demonstrates a reproducible benefit at the prespecified eight-evaluation checkpoint, including against default after 24 evaluations. Later search instability prevents a claim of uniformly faster convergence or reliable full-budget dominance.

## Question and scope

Does a body/hardware Young's-modulus initialization derived from the existing VLM material field improve fitting under a common model and computation budget? This pilot fits two shared parameters for every arm, so semantic initialization does not receive extra model capacity.

The 70-frame RGB-D case is synthetic: HAG4R generated its Genesis FEM trajectory using the same material asset that supplies the VLM prior. It is a prior-matched cross-simulator experiment, not independent evidence of VLM accuracy on real materials. Part labels are transferred from source FEM geometry; segmentation quality is outside this experiment. No physical Young's modulus can be inferred directly from the effective EMPM values below.

## Data and common model

- Input: `outputs/double_nailoong_grippers_20261011/final_data.pkl`, with 1765 tracked object points and 30 gripper tracks. Body: 1497 points; connector/ring/clasp: 70/107/91. There are no interior particles.
- Fit frames 1–63 from observed frame 0; hold out frames 64–69. Object-state predictions continue open-loop from frame 0, conditioned on the observed gripper motion, including during the held-out interval. The fitting objective is the mean of per-frame visible tracked-point squared distances, in world mm². Reported RMSE pools visible point-frame observations.
- Both trainable parameters are shared by all arms: weighted mean log E, and log(hardware E/body E). Both parameters remain free, including in homogeneous initializations.
- Same jelly material, nu=0.3, density=2000 kg/m³, particle volume=1e-5 in simulator units, and gravity=(0,0,-9.8). The primary refined study uses per-substep grid damping=sqrt(0.98); the earlier coarse study used 0.98.
- Frame duration=1/30 s, **334 substeps** in the refined study, substep duration=(1/30)/334 s. The earlier coarse study used 167 substeps. Grid32 over 0.064 simulator units; the existing coordinate transform divides physical lengths by five. Grid spacing 0.002 therefore corresponds to **10 mm physical**, not 2 mm.
- Common geometry correction: local-domain translation (0.54,0.47,0.496), ring/body anchors constrained to their corresponding parts, and nonoverlapping controller cubes with halfsize 0.003 simulator units (**15 mm physical**). These are kinematic grid-velocity boundaries, not resolved finger contact or squeezing.

The original default contact boxes covered 1754/1765 particles, with 1318 in both boxes; the original grid had only 17 occupied cells. Those settings cannot support a useful material comparison. The common corrections above were based on geometry, not on which initialization won. The sparse shell, missing volumetric connectivity, absent initial stress, and simplified contact remain important model errors.

## Initializations and fixed budget

Source VLM values are body 5 MPa and hardware 2.2 GPa (ratio 440). A literal forward diagnostic at the selected timestep was numerically incompatible with these GPa values. That is a solver/stability limitation, not evidence that the prior is wrong. The controlled study consequently uses one declared global multiplier, 2.3648224289674e-5, chosen to match the particle-weighted arithmetic mean E to the default 8000 Pa. This preserves contrast 440 but changes absolute material scale.

| Arm | Initialization |
|---|---|
| default | Body=hardware=8000 Pa |
| spatial | Body≈118.24 Pa; hardware≈52026.09 Pa |
| mean | Uniform particle-weighted geometric mean of spatial, approximately 298 Pa |
| reversed | Same weighted mean log E as spatial, with log contrast negated |

The reversed arm preserves weighted log mean and absolute log contrast, **not** the modulus histogram. Together, mean and reversed test whether any advantage depends on the spatial assignment rather than just a lower global scale. The main default/spatial comparison alone changes both scale and spatial pattern.

All arms use Nelder–Mead with the same two-parameter simplex increments (0.5 in each log parameter), seed 42, and **24 objective evaluations**. This is a finite-budget comparison, not a claim of convergence or of native EMPM optimizer performance. The complete 63-transition objective is evaluated using forward rollouts to avoid storing an unrolled adjoint on the 20-GiB GPU. Selection uses training loss only; test results are reported at budgets 0/8/16/24 without selecting a checkpoint by test error. Extra reporting rollouts are common to all arms and are not optimization evaluations.

Budget counts are optimizer candidate evaluations. Each arm also has a common initial full rollout and full reporting rollouts at the stated checkpoints. An eight-versus-24 evaluation comparison therefore does not mean an exact threefold wall-clock speedup.

## Refined-study results

These are the saved fresh-checkpoint pooled visible-point RMSEs, not square roots of cached optimizer losses. All values are in mm; every checkpoint was specified before this study ran.

| Initialization | Train @8 | Held-out @8 | Train @24 | Held-out @24 |
|---|---:|---:|---:|---:|
| Default | 14.206 | 21.852 | 13.333 | 18.281 |
| Spatial VLM | 11.230 | 12.695 | 5.433 | 8.123 |
| Same-log-mean uniform | 34.947 | 46.103 | 4.326* | 6.905* |
| Reversed contrast | 50.467 | 73.200 | 31.814* | 229.276* |

At eight evaluations, spatial initialization reduces training RMSE by **21.0%** and held-out RMSE by **41.9%** relative to default at the same budget. It also beats default at 24 evaluations by **15.8% / 30.6%**, respectively, and beats both spatial-assignment controls at eight evaluations. This is a specific limited-budget fitting benefit under a common two-parameter search. It is not a claim that the semantic prior is always a better starting prediction: its initial held-out RMSE is 80.30 mm, versus default's 44.29 mm.

The net benefit is concentrated in hardware. At budget 8, held-out ring RMSE falls from 22.896 to 3.066 mm and clasp RMSE from 97.492 to 28.413 mm; body RMSE increases from 6.924 to 12.097 mm. Thus the overall improvement does not mean every semantic part improves. Per-part counts and errors for every checkpoint are retained in `experiment_refined/metrics.csv`.

The initial coarse run motivated a common numerical refinement. Within the refined study, the early common training threshold (90% of default's initial MSE) is reached at evaluation 2 by default and evaluation 6 by spatial. Thus “faster convergence” without naming a target/budget would be misleading. The uniform control later catches up and has a lower saved final error than spatial. Its starred final checkpoint, however, differs from its recorded training MSE by 32.5%; reversed's differs by 17.3%. Spatial's checkpoint 16 is also numerically inconsistent: recorded training MSE is 41.95 mm², but its fresh checkpoint gives 74350.38 mm². These results remain in the tables and figures; unsafe whole-run incumbent AUCs cannot establish convergence speed.

At checkpoint 8, fitted spatial E has moved from 118.24/52026.09 Pa (body/hardware, 440:1) to 1676.67/271398.09 Pa (about 162:1). The result concerns an optimization initialization, not recovery of the literal VLM values. The global scale normalization and prior-matched synthetic source further limit the interpretation.

The repeat protocol fixes all four checkpoint-8 fields and all four checkpoint-24 fields, then runs two independent pairs of fresh 64-frame and 70-frame rollouts for each. It compares recorded training objectives, fresh prefixes, and held-out errors; no favorable replay is selected. These are repeated fixed-field forward evaluations, **not repeated optimization trials**. Results are saved to `experiment_refined/replay/replay.json`.

All checkpoint-8 replays agree with their recorded training MSE within **0.010%**; the largest train-only versus full-horizon prefix trajectory RMSE is **0.00095 mm**. Held-out RMSE across the two full replays is 21.8436–21.8477 mm (default), 12.69506–12.69517 mm (spatial), 46.10250–46.10252 mm (mean), and 73.19966–73.19971 mm (reversed). This variability is far smaller than the same-budget ranking gaps. Default at checkpoint 24 also reproduces: training MSE across fresh train/full prefixes is 178.4148–178.4285 mm² and held-out RMSE is 18.2710–18.2792 mm. Spatial at checkpoint 24 reproduces at 29.6952–29.7011 mm² and 8.1221–8.1224 mm.

Both starred final controls fail reproducibility. Mean's fresh training MSE spans 13.1443–27.7649 mm² and held-out RMSE spans 6.0089–9.5889 mm, with up to 4.421 mm disagreement between train-only and full-horizon prefixes. Reversed's training MSE spans 871.25–1015.71 mm², its fresh held-out RMSE is 37.655–38.278 mm instead of the saved 229.276 mm, and prefix disagreement reaches 10.187 mm. These ranges are retained as numerical failure evidence, not uncertainty intervals for a statistical estimate. No final ranking involving those fields is accepted.

![Prespecified eight-evaluation comparison with default at 24 as reference](../outputs/nailoong_material_init/comparison_refined/budget8_pooled_rmse.png)

![Recorded optimization curves and saved fresh-checkpoint discrepancies](../outputs/nailoong_material_init/comparison_refined/optimization_curves.png)

## Gradient repair

The initial autodiff pilot uncovered an incorrect Warp adjoint; its optimization results are excluded. The repaired original diagnostic now has AD=-0.005204251036 and FD=-0.005204556510, a relative error of 0.00587%, versus 97.894% before repair. Twenty-two directional checks cover E, nu, density, multiple observation times, moving boundaries, and heterogeneous materials. A separate mixed-direction/reversing-controller check also passes. See [the detailed gradient report](mpm_gradient_verification.md). The full comparison is rerun after the repair.

## Forward stability diagnosis and one common refinement

The initial repaired-code study (`experiment_full_fixed`) gave default/spatial fresh training RMSE 22.48/9.40 mm and held-out RMSE 38.37/12.76 mm. However, the mean control's optimizer recorded MSE 60.827 mm², while its fresh saved rollout collapsed at frame 7 and gave training RMSE 260.04 mm. Replaying the same mean field (body 13298.44 Pa, hardware 104396.23 Pa) again produced incompatible prefixes. This makes the four-arm attribution inconclusive; the apparently favorable default/spatial comparison alone cannot establish semantic benefit.

A code audit found no horizon-dependent forward branch or state-reset alias. At half the timestep, with square-root damping to preserve nominal attenuation per frame, two full replays of the problematic field gave training MSE 72.8732 and 72.8751 mm², with no particles clamped at the upper corner. This supports timestep instability as the explanation for the coarse-run failure. Evidence is saved in `replay_check.json` and `time_step_diagnostic.json` under `outputs/nailoong_material_init/`.

The single refined study restarts all four original initializations, keeps the split/objective/optimizer/24-evaluation budget/geometry fixed, and uses **334 substeps per frame**, timestep `(1/30)/334`, and per-substep damping `sqrt(0.98)=0.9899494936611666`. It does not start from any coarse-run fitted field. A checkpoint can support an interpretation only when its recorded training-prefix loss reproduces in fresh rollouts. Later unstable checkpoints are not used to infer convergence. No prior or numerical setting was retuned after observing the refined study's eight-evaluation result.

```bash
.conda/empm/bin/python scripts/train_test/run_nailoong_material_init.py \
  --arms default spatial mean reversed --start-frame 0 --train-end 64 --eval-end 70 \
  --max-evaluations 24 --checkpoints 0 8 16 24 \
  --vlm-scale 2.3648224289674e-05 --sim-dt .0001 \
  --grid-damping .9899494936611666 \
  --output-dir outputs/nailoong_material_init/experiment_refined
```

## Excitation diagnosis

The observed motion is dominated by rigid displacement. After removing a best rigid transform, shape RMS is 1.22 mm at frame 40 and 3.18 mm at frame 63. Simply holding the last training observation predicts the nearly stationary held-out tail with **1.049 mm RMSE**. That reference has access to the true frame-63 state, while the fitted model continues open-loop from frame 0; it diagnoses the weak held-out excitation rather than providing an equal-information forecast comparison. Frame-0 persistence is the corresponding initial-state-only reference: **6.184 mm training / 17.102 mm held-out RMSE**. Exact counts and values are saved in `outputs/nailoong_material_init/persistence_references.json`. Six adjacent test frames from one action do not establish generalization to a new excitation.

A separate, prespecified same-MPM diagnostic attempted a 15-mm axial pull/hold/return, with body/hardware truth fixed at 3000/30000 Pa before fitting. It was rejected as an informative recovery experiment: before the pull, some particles had already moved 33.65 mm; by frame 64 the maximum displacement was 224.95 mm and 60 particles exceeded 100 mm. Unsupported sparse hardware dominates the motion. No initialization comparison is reported on this unsuitable target, and its material truth was not retuned to favor the prior. Saved evidence: `outputs/nailoong_mpm_synthetic_recovery/diagnostic.json`.

The next useful excitation experiment requires connected volumetric geometry and represented grasp contact first, then an independently specified axial pull and held-out return, ideally regenerated through the source FEM/RGB-D pipeline. Merely increasing commanded motion on the current sparse model would not resolve this limitation.

## Reproduce

```bash
.conda/empm/bin/python scripts/data_process/prepare_nailoong_material_init.py
.conda/empm/bin/python scripts/eval/verify_mpm_gradients.py
.conda/empm/bin/python scripts/train_test/run_nailoong_material_init.py \
  --arms default spatial mean reversed --start-frame 0 --train-end 64 --eval-end 70 \
  --max-evaluations 24 --checkpoints 0 8 16 24 \
  --vlm-scale 2.3648224289674e-05 --sim-dt .0001 \
  --grid-damping .9899494936611666 \
  --output-dir outputs/nailoong_material_init/experiment_refined
.conda/empm/bin/python scripts/eval/summarize_nailoong_material_init.py \
  --run-dir outputs/nailoong_material_init/experiment_refined \
  --output-dir outputs/nailoong_material_init/comparison_refined
.conda/empm/bin/python scripts/eval/replay_nailoong_material_fit.py \
  --run-dir outputs/nailoong_material_init/experiment_refined \
  --evaluations 8 24 --repeats 2
```

The run saves configurations, search histories, per-part metrics, parameter fields, and prediction trajectories. The summarizer saves comparison/convergence/framewise CSVs and PNG/PDF figures. GPU floating-point reductions can cause small run-to-run differences; no statistical significance is inferred from this single trajectory.
