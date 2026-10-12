# Warp MPM gradient verification

Date: 2026-10-12. Environment: `.conda/empm`, Warp 1.8.1, CUDA float32 solver.

The repaired offline trajectory and endpoint interfaces agree with centered finite differences in 22 directional checks. At log-parameter step 0.01, every relative error is below 0.194%. Across all three steps (0.03, 0.01, 0.003), the largest error is 1.351%.

## Reproduce

```bash
.conda/empm/bin/python scripts/eval/verify_mpm_gradients.py
# Smaller isolated solver check:
.conda/empm/bin/python scripts/eval/verify_mpm_gradients.py --suite block
```

Full numerical results, including both perturbed losses, are saved to `outputs/nailoong_material_init/gradient_verification.json`. Use `--output` to preserve separate runs. Each finite-difference evaluation creates a fresh solver, particles, and controllers. The differentiation variable is a scalar log perturbation: vector fields use `E(q) = E(0) exp(q d)` or `rho(q) = rho(0) exp(q d)`; scalar E and nu use the same log perturbation. Smooth losses accumulate in float64 to reduce subtraction noise; the original diagnostic uses its float32 loss.

## Original failed diagnostic

The original frame-40 to frame-41 Nailoong diagnostic used uniform E=8000 Pa, nu=0.3, rho=2000, grid32 over 0.064, translation (0.54, 0.47, 0.496), and 167 substeps per 1/30-second frame. Its loss is 1000 times the sum of visible-point Chamfer distance and the Frobenius tracking norm on `object_motions_valid[41]`, divided by that mask count.

| Quantity | Before repair | After repair |
|---|---:|---:|
| Adjoint uniform log-E derivative | -0.0001096136475 | -0.0052042510360 |
| Centered FD, epsilon=0.01 | -0.0052045565099 | -0.0052045565099 |
| Relative derivative error | 97.894% | 0.00587% |

The finite-difference derivative reproduces the original value exactly. The original logged center loss was approximately 0.01201405935; the new run gives 0.01201385073, an absolute difference of 2.09e-7. That small forward drift is recorded rather than assigned a definite cause. Independent moving-controller probes before and after the boundary rewrite retained identical forward loss values.

## Repairs

- Sample each frame at its last substep and seed each returned position array with its own output cotangent. Do not sum temporal cotangents onto the final state.
- Seed Warp adjoints directly, avoiding a synthetic squared-position loss.
- Sum per-particle scalar-parameter gradients. Apply the 1e6 MPa-to-Pa chain rule for scalar E in the trajectory interface; the endpoint interface retains its existing Pa convention.
- Differentiate grid mass, including the velocity-normalization denominator. Share constant particle mass/density/volume arrays through the rollout so every step contributes to density derivatives.
- Snapshot Warp struct arguments at each taped launch, preserving each control step's geometry and velocity.
- Make grid boundary kernels read the incoming velocity and write a separate complete velocity array during differentiation. Overwritten contact velocities must not pass their cotangents to the incoming grid velocity.
- Execute the control callback in the endpoint interface as well as the trajectory interface.
- Align the native offline objective with the returned endpoints: simulated frame 1 matches observed frame 1, and the motion-validity mask remains the mask for transition 0 to 1. Use the same rounded frame/substep count for forward and optimization paths.

## Independent coordinator check

A separate 125-particle check used seed 31, a 10:1 heterogeneous field (3000/30000 Pa), mixed log-E direction (1/-0.4), two observation times with weights 0.4/1.7, and a controller whose x velocity reverses from 0.09 to -0.04. Its adjoint was 32.8432655. Centered differences at steps 0.02, 0.01, and 0.005 had relative errors 0.0164%, 0.0230%, and 0.0802%. The trajectory endpoint equaled the endpoint-interface output exactly; differentiable and ordinary forward trajectories also agreed exactly. Results are saved to `outputs/nailoong_material_init/gradient_independent.json`.

## Directional results

The block has 125 particles with a nonuniform initial velocity, three weighted observation times, and either no controller or a moving controller whose velocity changes during the rollout. The Nailoong test uses 1765 particles, active observed controllers, and distinct losses at frames 41 and 42. Its heterogeneous field has a particle-weighted mean of 8000 Pa, body E approximately 118.24 Pa and hardware E approximately 52026.09 Pa (440:1).

The table shows FD at epsilon=0.01 and the largest relative error over all three epsilon values. Names distinguish the scalar trajectory E in MPa from endpoint E in Pa.

| Check | Adjoint | FD (0.01) | Maximum relative error |
|---|---:|---:|---:|
| `block_no_controller/uniform_E` | -126.985527 | -127.050539 | 0.0536% |
| `block_no_controller/half_E` | -76.5130157 | -76.5256746 | 0.1693% |
| `block_no_controller/scalar_E_MPa` | -126.985619 | -127.045234 | 0.0587% |
| `block_no_controller/scalar_nu` | 14.6975832 | 14.7147243 | 0.1405% |
| `block_no_controller/uniform_rho` | 126.985481 | 127.047552 | 0.0573% |
| `block_no_controller/endpoint_scalar_E_Pa` | -38.3700333 | -38.3964926 | 0.0689% |
| `block_moving_controller/uniform_E` | 15.8562164 | 15.8807113 | 0.5996% |
| `block_moving_controller/half_E` | -12.2285624 | -12.2522997 | 1.3506% |
| `block_moving_controller/scalar_E_MPa` | 15.856246 | 15.8849162 | 0.4095% |
| `block_moving_controller/scalar_nu` | 24.1890316 | 24.1947406 | 0.1472% |
| `block_moving_controller/uniform_rho` | -15.8562355 | -15.8809762 | 0.4552% |
| `block_moving_controller/endpoint_scalar_E_Pa` | 9.97863007 | 9.98779627 | 0.3959% |
| `nailoong_controllers_uniform/uniform_E` | -0.631238222 | -0.631951326 | 0.1353% |
| `nailoong_controllers_uniform/body_E` | -0.441687733 | -0.441527643 | 0.0362% |
| `nailoong_controllers_uniform/hardware_E` | -0.18955043 | -0.189439607 | 0.4068% |
| `nailoong_controllers_uniform/scalar_E_MPa` | -0.631238401 | -0.631962568 | 0.1252% |
| `nailoong_controllers_uniform/scalar_nu` | -0.0508579202 | -0.0508404002 | 1.3098% |
| `nailoong_controllers_uniform/uniform_rho` | 0.631238222 | 0.631987175 | 0.1287% |
| `nailoong_controllers_uniform/original_endpoint_loss` | -0.00520425104 | -0.00520455651 | 0.0243% |
| `nailoong_controllers_heterogeneous/uniform_E` | -4.66751337 | -4.6669884 | 0.0236% |
| `nailoong_controllers_heterogeneous/body_E` | -4.31725502 | -4.31687223 | 0.0173% |
| `nailoong_controllers_heterogeneous/hardware_E` | -0.350257963 | -0.350475513 | 0.3496% |

## Scope

These checks cover `MPMDifferentiableSimulationTrail` and `MPMDifferentiableSimulation` for elastic jelly with grid boundary conditions, scalar and spatial E, scalar nu, and density directions. They do not qualify the unused `WCheckpoint`/`Clean` interfaces, plastic materials, static-position reconstruction, or warmup truncation. Contact/clamping branches are piecewise smooth; finite differences that cross a branch need not equal a local adjoint. The controller-free Nailoong window is nearly rigid free fall and has negligible material sensitivity, so the strained block supplies the informative no-controller check.
