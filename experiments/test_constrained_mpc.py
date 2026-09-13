"""Lightweight checks for the constrained-MPC experiment."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebmul, chebval

from run_constrained_mpc import (
    benchmark_plants,
    collect_reference_systems,
    cost_gated_solver,
    execute_certificate,
    execute_cg,
    execute_direct,
    execute_interval_chebyshev,
    condensed_problem,
    solve_end_to_end_qp,
    state_panel,
    validate_systems,
)


def test_state_panel_is_nonantipodal_and_has_fixed_radius() -> None:
    states = state_panel()
    assert len(states) == 8
    assert np.allclose([np.linalg.norm(state) for state in states], 0.75)
    normalized = np.asarray(states) / 0.75
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            assert not np.isclose(abs(float(normalized[left] @ normalized[right])), 1.0)


def test_weyl_plans_and_all_solver_residuals() -> None:
    plant = benchmark_plants()[0]
    systems, qp = collect_reference_systems(
        plant=plant,
        horizon=8,
        state_index=0,
        state=state_panel()[0],
        barrier_weights=(0.1,),
        tolerance=0.2,
        maximum_iterations=30,
    )
    assert qp["reference_converged"]
    assert systems
    methods = {
        "certificate": execute_certificate,
        "cost_gated_certificate": cost_gated_solver(22),
        "interval_chebyshev": execute_interval_chebyshev,
        "cg": execute_cg,
        "direct": execute_direct,
    }
    validation = validate_systems(systems[:4], methods)
    assert len(validation) == 4 * len(methods)
    for system in systems[:4]:
        plan = system.plan
        scaled_zero = -plan.center / plan.radius
        assert np.isclose(
            chebval(scaled_zero, plan.residual_coefficients),
            1.0,
            atol=2.0e-10,
        )
        numerator = -plan.residual_coefficients.copy()
        numerator[0] += 1.0
        reconstructed = chebmul(
            plan.correction_coefficients,
            np.asarray([plan.center, plan.radius]),
        )
        assert np.allclose(
            reconstructed,
            numerator[: reconstructed.size],
            atol=2.0e-9,
            rtol=2.0e-9,
        )


def test_certificate_end_to_end_barrier_solve() -> None:
    plant = benchmark_plants()[0]
    hessian, gradient_map = condensed_problem(plant, horizon=8)
    eigenvalues = np.linalg.eigvalsh(hessian)
    result = solve_end_to_end_qp(
        hessian=hessian,
        linear_term=gradient_map @ state_panel()[1],
        input_bound=plant.input_bound,
        base_interval=(float(eigenvalues[0]), float(eigenvalues[-1])),
        barrier_weights=(0.1,),
        tolerance=0.2,
        method_name="certificate",
        method=execute_certificate,
        maximum_iterations=30,
    )
    assert result["converged"]
    assert result["linear_systems"] > 0
    assert result["matrix_products"] > 0
    assert result["maximum_linear_relative_residual"] <= 0.2 * (1.0 + 3.0e-6)
    assert result["feasibility_violation"] == 0.0
