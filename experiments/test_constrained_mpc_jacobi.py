"""Lightweight checks for the exploratory fixed-Jacobi runner."""

from __future__ import annotations

import run_constrained_mpc as base
from run_constrained_mpc_jacobi_pilot import (
    method_panel,
    original_residual_rows,
    transform_configuration,
)


def test_fixed_jacobi_weyl_enclosure_and_residuals() -> None:
    plant = base.benchmark_plants()[0]
    (
        _hessian,
        _gradient_map,
        scale,
        _base_interval,
        transformed,
        raw,
        qps,
    ) = transform_configuration(
        plant=plant,
        horizon=8,
        states=base.state_panel()[:1],
        barrier_weights=(0.1,),
        tolerance=0.2,
    )
    assert qps[0]["reference_converged"]
    assert transformed
    assert all(
        system.plan.lower <= system.exact_lower * (1.0 + 1.0e-12)
        and system.plan.upper >= system.exact_upper * (1.0 - 1.0e-12)
        for system in transformed
    )
    methods = method_panel(scale, gate_degree=22)
    rows = original_residual_rows(transformed[:3], raw[:3], scale, methods)
    assert len(rows) == 3 * len(methods)
    assert all(
        float(row["preconditioned_relative_residual"]) <= 0.2 * (1.0 + 3.0e-6)
        for row in rows
        if row["method"] != "direct"
    )
    accepted = [
        row
        for row in rows
        if row["method"] == "certificate" and row["accepted_degree_two"]
    ]
    assert all(
        float(row["original_relative_residual"]) <= 0.2 * (1.0 + 3.0e-6)
        for row in accepted
    )
