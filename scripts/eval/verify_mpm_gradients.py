"""Check Warp/PyTorch material adjoints against centered finite differences.

Run with .conda/empm/bin/python scripts/eval/verify_mpm_gradients.py.
The small strained block isolates the solver; Nailoong exercises the actual
moving controllers and frame-40 observations. All finite-difference rollouts
start from fresh states and use the same forward interface as the adjoint.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import warp as wp

from phys_sim.diff_simulator.warp_mpm.interface import (
    MPMDifferentiableSimulation,
    MPMDifferentiableSimulationTrail,
)
from phys_sim.diff_simulator.warp_mpm.warp_mpm_wrapper import WarpMPMWrapper


def evaluate(factory, objective, direction, parameter, q_value, frames,
             substeps, dt, differentiable, endpoint=False):
    mpm = factory(differentiable)
    q = torch.tensor(q_value, device=mpm.device, requires_grad=differentiable)
    E, nu, rho = mpm.E_t.detach(), mpm.nu_t.detach(), mpm.rho_t.detach()
    if parameter == "E":
        E = E * torch.exp(q * direction)
    elif parameter == "scalar_E":
        E = torch.tensor(8000.0 if endpoint else .008, device=mpm.device) * torch.exp(q)
    elif parameter == "scalar_nu":
        nu = torch.tensor(.3, device=mpm.device) * torch.exp(q)
    elif parameter == "rho":
        rho = rho * torch.exp(q * direction)
    interface = MPMDifferentiableSimulation if endpoint else MPMDifferentiableSimulationTrail
    positions = interface.apply(
        mpm.mpm_solver, mpm.mpm_state, mpm.mpm_model,
        substeps, dt, frames * substeps, mpm.x_t, mpm.v_t,
        E, nu, rho, torch.ones(mpm.n_particles, dtype=torch.bool, device=mpm.device),
        None, mpm.device, differentiable, 0,
    )
    if endpoint:
        positions = positions[None]
    loss = objective(positions.double())
    derivative = float(torch.autograd.grad(loss, q)[0]) if differentiable else None
    return float(loss.detach()), derivative, positions.detach()


def check(name, factory, objective, direction, parameter, frames, substeps, dt,
          epsilons, endpoint=False):
    args = (factory, objective, direction, parameter)
    kwargs = dict(frames=frames, substeps=substeps, dt=dt, endpoint=endpoint)
    loss, adjoint, _ = evaluate(*args, 0.0, differentiable=True, **kwargs)
    differences = []
    for epsilon in epsilons:
        plus = evaluate(*args, epsilon, differentiable=False, **kwargs)[0]
        minus = evaluate(*args, -epsilon, differentiable=False, **kwargs)[0]
        derivative = (plus - minus) / (2 * epsilon)
        error = abs(adjoint - derivative) / max(abs(adjoint), abs(derivative), 1e-12)
        differences.append(dict(epsilon=epsilon, derivative=derivative,
                                relative_error=error, absolute_error=abs(adjoint - derivative),
                                plus_loss=plus, minus_loss=minus))
    result = dict(name=name, parameter=parameter, frames=frames, substeps=substeps,
                  dt=dt, loss=loss, adjoint=adjoint, finite_differences=differences)
    print(json.dumps(result), flush=True)
    return result


def block_checks(epsilons):
    torch.manual_seed(17)
    x = torch.cartesian_prod(*[torch.linspace(.021, .043, 5, device="cuda:0")] * 3)
    x += torch.rand_like(x) * .00015
    v = (x - .032) @ torch.tensor([[12., 3., 0.], [1., -5., 1.], [0., 2., -2.]], device=x.device)
    weight = ((x - .032) * torch.tensor([1., -.8, .5], device=x.device)).double()
    results = []
    for controlled in (False, True):
        def factory(grad):
            mpm = WarpMPMWrapper(grid_size=24, grid_lim=.064, requires_grad=grad,
                                 sim_dt=.0002, frame_dt=.008)
            mpm.gravity = [0., 0., 0.]
            mpm.add_particles(x, v=v, E=8000.)
            mpm.initialize()
            if controlled:
                mpm.mpm_solver.set_grid_velocity_on_geometry(dict(
                    shape="cube", center=[.020, .032, .032], size=[.005, .016, .016],
                    velocity=[.12, .03, 0.]))

                def callback(step):
                    speed = .12 if step < 80 else .16
                    mpm.mpm_solver.collider_params[0].velocity = wp.vec3(speed, .03, 0.)

                mpm.mpm_solver.step_callback = callback
            return mpm

        # Unequal weights expose accidental placement of all cotangents at the
        # last frame. Signed spatial weights retain material sensitivity.
        time_weights = torch.tensor([.5, 1., 2.], device=x.device, dtype=torch.float64)

        def objective(positions):
            terms = ((positions - x.double()) * weight).sum(dim=(1, 2))
            return (terms * time_weights).sum() * 1e5

        prefix = "block_moving_controller" if controlled else "block_no_controller"
        directions = (("uniform_E", "E", torch.ones(len(x), device=x.device)),
                      ("half_E", "E", (x[:, 0] > .032).float()),
                      ("scalar_E_MPa", "scalar_E", None),
                      ("scalar_nu", "scalar_nu", None),
                      ("uniform_rho", "rho", torch.ones(len(x), device=x.device)))
        for suffix, parameter, direction in directions:
            results.append(check(prefix + "/" + suffix, factory, objective, direction,
                                 parameter, 3, 40, .0002, epsilons))
        # Both interfaces must return the same final frame and control schedule.
        endpoint_objective = lambda positions: ((positions - x.double()) * weight).sum() * 1e5
        results.append(check(prefix + "/endpoint_scalar_E_Pa", factory, endpoint_objective,
                             None, "scalar_E", 1, 120, .0002, epsilons, endpoint=True))
    return results


def nailoong_checks(epsilons):
    from phys_sim import InvPhyTrainerWarpMPM
    from phys_sim.engine.utils import transform_points
    from phys_sim.utils import cfg
    from pytorch3d.loss import chamfer_distance
    from scripts.train_test.run_nailoong_material_init import configure_controllers, make_mpm

    args = SimpleNamespace(case_dir=Path("outputs/double_nailoong_grippers_20261011"),
                           start_frame=40, frame_dt=1 / 30, controller_halfsize=.003,
                           controller_vscale=None, controller_anchor="semantic")
    cfg.load_from_yaml("configs/mpm_config.yaml")
    cfg.case_name = args.case_dir.name
    field = np.load("outputs/nailoong_material_init/field.npz")
    labels = field["part_labels"]
    trainer = InvPhyTrainerWarpMPM(
        data_path=str(args.case_dir / "final_data.pkl"),
        base_dir="outputs/nailoong_material_init/gradient_checks", train_frame=44,
        save_artifacts=False, mpm_kwargs=dict(grid_size=32, grid_lim=.064,
                                             frame_dt=1 / 30, sim_dt=(1 / 30) / 167),
        mpm_shift=(.54, .47, .496))
    configure_controllers(trainer, labels, args)
    target = transform_points(trainer.object_points[41:43].clone(), name=trainer.exp_name)
    target -= torch.as_tensor(trainer.mpm_shift, device=target.device)
    target = target.double()
    results = []
    # A zero-velocity, controller-free Nailoong window is almost rigid free fall
    # and has negligible E sensitivity. The strained block supplies the
    # informative no-controller check instead.
    raw_E = field["youngs_modulus_pa"].astype(np.float32)
    fields = (("uniform", np.full(len(labels), 8000., dtype=np.float32)),
              ("heterogeneous", raw_E * (8000. / raw_E.mean(dtype=np.float64))))
    for field_name, base_E in fields:
        def factory(grad):
            mpm = make_mpm(trainer, base_E, args, grad)
            mpm.mpm_solver.step_callback = lambda step: mpm.frame_callback(mpm, step // 167)
            return mpm

        def temporal_loss(positions):
            per_frame = ((positions - target) ** 2).sum(dim=2).mean(dim=1)
            return (per_frame * torch.tensor([.7, 1.3], device=target.device)).sum() * 1e6

        prefix = "nailoong_controllers_" + field_name
        directions = [
            ("uniform_E", "E", torch.ones(len(labels), device=target.device)),
            ("body_E", "E", torch.tensor(labels == 0, device=target.device).float()),
            ("hardware_E", "E", torch.tensor(labels != 0, device=target.device).float()),
        ]
        if field_name == "uniform":
            directions.extend([
            ("scalar_E_MPa", "scalar_E", None),
            ("scalar_nu", "scalar_nu", None),
            ("uniform_rho", "rho", torch.ones(len(labels), device=target.device)),
            ])
        for suffix, parameter, direction in directions:
            results.append(check(prefix + "/" + suffix, factory, temporal_loss, direction,
                                 parameter, 2, 167, (1 / 30) / 167, epsilons))
        if field_name == "uniform":
            def legacy_endpoint_loss(positions):
                # Match the loss used by the original failed endpoint probe.
                visible = trainer.object_visibilities[41]
                valid = trainer.object_motions_valid[41]
                points = positions.float()
                observed = target[:1].float()
                return (chamfer_distance(points[:, visible], observed[:, visible])[0]
                        + (points[:, valid] - observed[:, valid]).norm() / valid.sum()) * 1000

            results.append(check(prefix + "/original_endpoint_loss", factory, legacy_endpoint_loss,
                                 torch.ones(len(labels), device=target.device), "E",
                                 1, 167, (1 / 30) / 167, epsilons))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "block", "nailoong"), default="all")
    parser.add_argument("--epsilons", nargs="+", type=float, default=(.03, .01, .003))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/nailoong_material_init/gradient_verification.json"))
    args = parser.parse_args()
    wp.init()
    results = []
    if args.suite in ("all", "block"):
        results.extend(block_checks(args.epsilons))
    if args.suite in ("all", "nailoong"):
        results.extend(nailoong_checks(args.epsilons))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(warp_version=wp.__version__,
                                         dtype="float32 forward; float64 smooth loss; float32 original endpoint loss",
                                         perturbation="log parameter",
                                         max_relative_error_by_epsilon={
                                             str(eps): max(r["finite_differences"][i]["relative_error"] for r in results)
                                             for i, eps in enumerate(args.epsilons)},
                                         results=results), indent=2) + "\n")
    print(f"Saved {len(results)} directional checks to {args.output}", flush=True)


if __name__ == "__main__":
    main()
