"""Fair linear-solver replay on box-constrained MPC Newton systems.

Two standard linear plants (a double integrator and a damped oscillator) are
condensed over several horizons.  Input-box constraints are handled with a
logarithmic barrier.  Its Newton matrix is

    A(u, nu) = H + diag(nu / (umax-u)^2 + nu / (umax+u)^2),

and therefore changes with the barrier iterate and continuation parameter.
The matrices and right-hand sides are collected along reference damped-Newton
paths, then replayed identically for every linear solver.

The comparison is deliberately favorable to interval Chebyshev.  Eigenvalue
bounds for the fixed condensed Hessian H are computed once per configuration.
For every Newton system, Weyl's inequalities update those bounds using only
the minimum and maximum barrier diagonal.  The resulting Chebyshev degree,
residual coefficients, correction coefficients, and scale are all prepared
before timing any solver.  Timed code contains only the actual linear solve.
No artificial latency is inserted.

This is a Newton-system replay, not an end-to-end comparison of constrained
MPC controllers.  Direct factorization and adaptive CG are included to expose
the scope of any benefit from a predetermined polynomial schedule.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import numpy as np
from numpy.polynomial.chebyshev import chebmul, chebval
from scipy.linalg import expm

from fmcert import (
    chebyshev_budget_for_tolerance,
    correction_chebyshev_coefficients,
)


Array = np.ndarray


@dataclass(frozen=True)
class Plant:
    name: str
    dynamics: Array
    control: Array
    state_weight: Array
    input_weight: Array
    input_bound: float


@dataclass(frozen=True)
class FixedPlan:
    lower: float
    upper: float
    tolerance: float
    center: float
    radius: float
    degree: int
    residual_coefficients: Array
    correction_coefficients: Array
    degree_two_evaluation: Array


@dataclass(frozen=True)
class NewtonSystem:
    plant: str
    horizon: int
    state_index: int
    barrier_stage: int
    newton_iteration: int
    barrier_weight: float
    matrix: Array
    gradient: Array
    plan: FixedPlan
    exact_lower: float
    exact_upper: float
    maximum_control_fraction: float
    newton_decrement_half: float


def discretize(continuous_a: Array, continuous_b: Array, step: float) -> tuple[Array, Array]:
    state_dimension, input_dimension = continuous_b.shape
    block = np.zeros(
        (state_dimension + input_dimension, state_dimension + input_dimension)
    )
    block[:state_dimension, :state_dimension] = continuous_a
    block[:state_dimension, state_dimension:] = continuous_b
    mapped = expm(step * block)
    return mapped[:state_dimension, :state_dimension], mapped[:state_dimension, state_dimension:]


def benchmark_plants() -> list[Plant]:
    step = 0.1
    double_integrator_a = np.asarray([[1.0, step], [0.0, 1.0]])
    double_integrator_b = np.asarray([[0.5 * step * step], [step]])
    oscillator_a, oscillator_b = discretize(
        np.asarray([[0.0, 1.0], [-2.0, -0.35]]),
        np.asarray([[0.0], [1.0]]),
        step,
    )
    return [
        Plant(
            name="double_integrator",
            dynamics=double_integrator_a,
            control=double_integrator_b,
            state_weight=np.diag([10.0, 1.0]),
            input_weight=np.asarray([[0.02]]),
            input_bound=1.0,
        ),
        Plant(
            name="damped_oscillator",
            dynamics=oscillator_a,
            control=oscillator_b,
            state_weight=np.diag([8.0, 1.0]),
            input_weight=np.asarray([[0.03]]),
            input_bound=1.0,
        ),
    ]


def condensed_problem(plant: Plant, horizon: int) -> tuple[Array, Array]:
    state_dimension, input_dimension = plant.control.shape
    state_map = np.zeros((state_dimension * horizon, state_dimension))
    input_map = np.zeros(
        (state_dimension * horizon, input_dimension * horizon)
    )
    for row in range(horizon):
        state_map[
            state_dimension * row : state_dimension * (row + 1)
        ] = np.linalg.matrix_power(plant.dynamics, row + 1)
        for column in range(row + 1):
            input_map[
                state_dimension * row : state_dimension * (row + 1),
                input_dimension * column : input_dimension * (column + 1),
            ] = (
                np.linalg.matrix_power(plant.dynamics, row - column)
                @ plant.control
            )
    block_state_weight = np.kron(np.eye(horizon), plant.state_weight)
    block_input_weight = np.kron(np.eye(horizon), plant.input_weight)
    hessian = (
        block_input_weight + input_map.T @ block_state_weight @ input_map
    )
    gradient_map = input_map.T @ block_state_weight @ state_map
    return hessian, gradient_map


def state_panel(
    *,
    count: int = 8,
    angle_offset: float = 0.0,
    radius: float = 0.75,
) -> list[Array]:
    """Return a predeclared non-antipodal two-state panel on a semicircle."""

    if count < 1 or radius <= 0.0:
        raise ValueError("state count and radius must be positive")
    angles = angle_offset + np.linspace(0.0, np.pi, count, endpoint=False)
    return [
        float(radius) * np.asarray([np.cos(angle), np.sin(angle)])
        for angle in angles
    ]


def barrier_terms(control: Array, bound: float, weight: float) -> tuple[float, Array, Array]:
    upper_slack = bound - control
    lower_slack = bound + control
    if np.any(upper_slack <= 0.0) or np.any(lower_slack <= 0.0):
        return (
            float("inf"),
            np.full_like(control, np.nan),
            np.full_like(control, np.nan),
        )
    value = -weight * float(
        np.sum(np.log(upper_slack) + np.log(lower_slack))
    )
    gradient = weight * (1.0 / upper_slack - 1.0 / lower_slack)
    diagonal = weight * (
        1.0 / upper_slack**2 + 1.0 / lower_slack**2
    )
    return value, gradient, diagonal


def barrier_objective(
    control: Array,
    hessian: Array,
    linear_term: Array,
    bound: float,
    weight: float,
) -> float:
    barrier_value, _, _ = barrier_terms(control, bound, weight)
    return (
        0.5 * float(control @ hessian @ control)
        + float(linear_term @ control)
        + barrier_value
    )


def maximum_interior_step(control: Array, direction: Array, bound: float) -> float:
    candidates = [1.0]
    positive = direction > 0.0
    negative = direction < 0.0
    if np.any(positive):
        candidates.append(
            float(np.min((bound - control[positive]) / direction[positive]))
        )
    if np.any(negative):
        candidates.append(
            float(np.min((-bound - control[negative]) / direction[negative]))
        )
    return min(candidates)


def make_fixed_plan(
    interval: tuple[float, float], tolerance: float
) -> FixedPlan:
    lower, upper = (float(value) for value in interval)
    if not (0.0 < lower < upper and 0.0 < tolerance < 1.0):
        raise ValueError("invalid interval or tolerance")
    degree = chebyshev_budget_for_tolerance(interval, tolerance)
    if degree < 2:
        raise ValueError("the constrained-MPC experiment requires degree at least two")
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    scaled_zero = -center / radius
    # Construct the one-sparse Chebyshev residual in O(N) memory.  The
    # library reference helper intentionally favors clarity and obtains a
    # basis vector from an identity matrix, which is unsuitable at large N.
    residual = np.zeros(degree + 1)
    residual[-1] = 1.0
    residual[-1] /= float(chebval(scaled_zero, residual))
    residual[0] += 1.0 - float(chebval(scaled_zero, residual))
    correction = correction_chebyshev_coefficients(residual, interval)
    evaluation = np.asarray(
        [1.0, scaled_zero, 2.0 * scaled_zero * scaled_zero - 1.0]
    )
    return FixedPlan(
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        center=center,
        radius=radius,
        degree=degree,
        residual_coefficients=residual,
        correction_coefficients=correction,
        degree_two_evaluation=evaluation,
    )


def scaled_product(matrix: Array, vector: Array, plan: FixedPlan) -> Array:
    return (matrix @ vector - plan.center * vector) / plan.radius


def execute_interval_chebyshev(
    matrix: Array, gradient: Array, plan: FixedPlan
) -> tuple[Array, Array, int, bool]:
    correction = plan.correction_coefficients
    residual_coefficients = plan.residual_coefficients
    previous = gradient
    point_polynomial = float(correction[0]) * previous
    residual = float(residual_coefficients[0]) * previous
    current = scaled_product(matrix, previous, plan)
    if correction.size > 1:
        point_polynomial = point_polynomial + float(correction[1]) * current
    if residual_coefficients[1] != 0.0:
        residual = residual + float(residual_coefficients[1]) * current
    for order in range(2, plan.degree + 1):
        following = 2.0 * scaled_product(matrix, current, plan) - previous
        if order < correction.size:
            point_polynomial = (
                point_polynomial + float(correction[order]) * following
            )
        if residual_coefficients[order] != 0.0:
            residual = residual + float(residual_coefficients[order]) * following
        previous, current = current, following
    return -point_polynomial, residual, plan.degree, False


def degree_two_correction(coefficients: Array, plan: FixedPlan) -> Array:
    constant, linear, quadratic = (float(value) for value in coefficients)
    correction_linear = -2.0 * quadratic / plan.radius
    correction_constant = (
        -linear - plan.center * correction_linear
    ) / plan.radius
    mismatch = abs(
        plan.center * correction_constant
        + 0.5 * plan.radius * correction_linear
        - (1.0 - constant)
    )
    if mismatch > 5.0e-9 * max(
        1.0, abs(constant), abs(linear), abs(quadratic)
    ):
        raise RuntimeError("degree-two correction lost normalization")
    return np.asarray([correction_constant, correction_linear])


def execute_certificate(
    matrix: Array, gradient: Array, plan: FixedPlan
) -> tuple[Array, Array, int, bool]:
    """Execute the exact degree-two test, reusing its probe on fallback."""

    zeroth = gradient
    first = scaled_product(matrix, zeroth, plan)
    second = 2.0 * scaled_product(matrix, first, plan) - zeroth
    mass = float(zeroth @ zeroth)
    gram = np.asarray(
        [
            [mass, float(zeroth @ first), float(zeroth @ second)],
            [float(first @ zeroth), float(first @ first), float(first @ second)],
            [float(second @ zeroth), float(second @ first), float(second @ second)],
        ]
    ) / mass
    try:
        inverse_evaluation = np.linalg.solve(
            gram, plan.degree_two_evaluation
        )
        coefficients = inverse_evaluation / float(
            plan.degree_two_evaluation @ inverse_evaluation
        )
        scaled_zero = -plan.center / plan.radius
        coefficients[0] += 1.0 - float(chebval(scaled_zero, coefficients))
        squared = chebmul(coefficients, coefficients)
        arithmetic_margin = 128.0 * np.finfo(float).eps * (
            1.0 + float(np.sum(np.abs(squared)))
        )
        certificate = (
            float(coefficients @ gram @ coefficients) + arithmetic_margin
        )
        accepted = bool(
            np.isfinite(certificate)
            and certificate <= plan.tolerance * plan.tolerance
        )
    except np.linalg.LinAlgError:
        coefficients = np.zeros(3)
        accepted = False

    if accepted:
        correction = degree_two_correction(coefficients, plan)
        point = -(
            float(correction[0]) * zeroth + float(correction[1]) * first
        )
        residual = (
            float(coefficients[0]) * zeroth
            + float(coefficients[1]) * first
            + float(coefficients[2]) * second
        )
        return point, residual, 2, True

    correction = plan.correction_coefficients
    residual_coefficients = plan.residual_coefficients
    point_polynomial = float(correction[0]) * zeroth
    residual = float(residual_coefficients[0]) * zeroth
    if correction.size > 1:
        point_polynomial = point_polynomial + float(correction[1]) * first
    if residual_coefficients[1] != 0.0:
        residual = residual + float(residual_coefficients[1]) * first
    if correction.size > 2:
        point_polynomial = point_polynomial + float(correction[2]) * second
    if residual_coefficients[2] != 0.0:
        residual = residual + float(residual_coefficients[2]) * second
    previous, current = first, second
    for order in range(3, plan.degree + 1):
        following = 2.0 * scaled_product(matrix, current, plan) - previous
        if order < correction.size:
            point_polynomial = (
                point_polynomial + float(correction[order]) * following
            )
        if residual_coefficients[order] != 0.0:
            residual = residual + float(residual_coefficients[order]) * following
        previous, current = current, following
    return -point_polynomial, residual, plan.degree, False


def execute_cg(
    matrix: Array, gradient: Array, plan: FixedPlan
) -> tuple[Array, Array, int, bool]:
    right_hand_side = -gradient
    point = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    initial_norm = float(np.linalg.norm(residual))
    residual_squared = float(residual @ residual)
    for products in range(1, plan.degree + 1):
        image = matrix @ direction
        curvature = float(direction @ image)
        step = residual_squared / curvature
        point += step * direction
        residual -= step * image
        next_squared = float(residual @ residual)
        if np.sqrt(next_squared) <= plan.tolerance * initial_norm:
            return point, -residual, products, False
        direction = residual + (next_squared / residual_squared) * direction
        residual_squared = next_squared
    return point, -residual, plan.degree, False


def execute_direct(
    matrix: Array, gradient: Array, _plan: FixedPlan
) -> tuple[Array, Array, int, bool]:
    point = np.linalg.solve(matrix, -gradient)
    return point, np.zeros_like(point), 0, False


def cost_gated_solver(
    minimum_degree: int,
) -> Callable[[Array, Array, FixedPlan], tuple[Array, Array, int, bool]]:
    def solve(
        matrix: Array, gradient: Array, plan: FixedPlan
    ) -> tuple[Array, Array, int, bool]:
        if plan.degree < minimum_degree:
            return execute_interval_chebyshev(matrix, gradient, plan)
        return execute_certificate(matrix, gradient, plan)

    return solve


def collect_reference_systems(
    *,
    plant: Plant,
    horizon: int,
    state_index: int,
    state: Array,
    barrier_weights: tuple[float, ...],
    tolerance: float,
    maximum_iterations: int = 80,
    precomputed_hessian: Array | None = None,
    precomputed_gradient_map: Array | None = None,
    precomputed_base_interval: tuple[float, float] | None = None,
) -> tuple[list[NewtonSystem], dict[str, object]]:
    if precomputed_hessian is None or precomputed_gradient_map is None:
        if not (
            precomputed_hessian is None and precomputed_gradient_map is None
        ):
            raise ValueError("precomputed Hessian and gradient map must be paired")
        hessian, gradient_map = condensed_problem(plant, horizon)
    else:
        hessian = np.asarray(precomputed_hessian)
        gradient_map = np.asarray(precomputed_gradient_map)
    if precomputed_base_interval is None:
        base_eigenvalues = np.linalg.eigvalsh(hessian)
        base_lower = float(base_eigenvalues[0])
        base_upper = float(base_eigenvalues[-1])
    else:
        base_lower, base_upper = (
            float(value) for value in precomputed_base_interval
        )
    linear_term = gradient_map @ state
    control = np.zeros(horizon * plant.control.shape[1])
    systems: list[NewtonSystem] = []
    converged = True
    for stage, barrier_weight in enumerate(barrier_weights):
        for iteration in range(maximum_iterations):
            _, barrier_gradient, barrier_diagonal = barrier_terms(
                control, plant.input_bound, barrier_weight
            )
            gradient = hessian @ control + linear_term + barrier_gradient
            matrix = hessian + np.diag(barrier_diagonal)
            direction = np.linalg.solve(matrix, -gradient)
            decrement_half = -0.5 * float(gradient @ direction)
            objective_value = barrier_objective(
                control,
                hessian,
                linear_term,
                plant.input_bound,
                barrier_weight,
            )
            if decrement_half <= 1.0e-11 * max(1.0, abs(objective_value)):
                break
            exact_eigenvalues = np.linalg.eigvalsh(matrix)
            weyl_interval = (
                base_lower + float(np.min(barrier_diagonal)),
                base_upper + float(np.max(barrier_diagonal)),
            )
            plan = make_fixed_plan(weyl_interval, tolerance)
            systems.append(
                NewtonSystem(
                    plant=plant.name,
                    horizon=horizon,
                    state_index=state_index,
                    barrier_stage=stage,
                    newton_iteration=iteration,
                    barrier_weight=barrier_weight,
                    matrix=matrix,
                    gradient=gradient,
                    plan=plan,
                    exact_lower=float(exact_eigenvalues[0]),
                    exact_upper=float(exact_eigenvalues[-1]),
                    maximum_control_fraction=float(
                        np.max(np.abs(control)) / plant.input_bound
                    ),
                    newton_decrement_half=decrement_half,
                )
            )
            step = 0.995 * maximum_interior_step(
                control, direction, plant.input_bound
            )
            slope = float(gradient @ direction)
            while barrier_objective(
                control + step * direction,
                hessian,
                linear_term,
                plant.input_bound,
                barrier_weight,
            ) > objective_value + 1.0e-4 * step * slope:
                step *= 0.5
                if step < 2.0**-45:
                    raise RuntimeError("reference Newton line search failed")
            control = control + step * direction
        else:
            converged = False
    return systems, {
        "plant": plant.name,
        "horizon": horizon,
        "state_index": state_index,
        "state_0": float(state[0]),
        "state_1": float(state[1]),
        "reference_converged": converged,
        "newton_systems": len(systems),
        "final_maximum_control_fraction": float(
            np.max(np.abs(control)) / plant.input_bound
        ),
    }


def validate_systems(
    systems: list[NewtonSystem],
    methods: dict[str, Callable[[Array, Array, FixedPlan], tuple[Array, Array, int, bool]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for system_index, system in enumerate(systems):
        normalizer = float(np.linalg.norm(system.gradient))
        if system.plan.lower > system.exact_lower * (1.0 + 2.0e-11):
            raise RuntimeError("Weyl lower endpoint exceeds the exact eigenvalue")
        if system.plan.upper < system.exact_upper * (1.0 - 2.0e-11):
            raise RuntimeError("Weyl upper endpoint is below the exact eigenvalue")
        for method_name, method in methods.items():
            point, reported_residual, products, accepted = method(
                system.matrix, system.gradient, system.plan
            )
            direct_residual = system.matrix @ point + system.gradient
            relative_residual = float(np.linalg.norm(direct_residual)) / normalizer
            if method_name != "direct":
                if relative_residual > system.plan.tolerance * (1.0 + 2.0e-6):
                    raise RuntimeError(
                        f"{method_name} missed the residual target on system "
                        f"{system_index}: {relative_residual}"
                    )
                mismatch = float(
                    np.linalg.norm(direct_residual - reported_residual)
                ) / normalizer
                if mismatch > 2.0e-7:
                    raise RuntimeError(
                        f"{method_name} residual mismatch on system "
                        f"{system_index}: {mismatch}"
                    )
            rows.append(
                {
                    "system": system_index,
                    "method": method_name,
                    "products": products,
                    "accepted_degree_two": accepted,
                    "relative_residual": relative_residual,
                }
            )
    return rows


def time_systems(
    systems: list[NewtonSystem],
    methods: dict[str, Callable[[Array, Array, FixedPlan], tuple[Array, Array, int, bool]]],
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    for method in methods.values():
        for system in systems[: min(3, len(systems))]:
            method(system.matrix, system.gradient, system.plan)
    jobs = [
        (repeat, system_index, method_name)
        for repeat in range(repeats)
        for system_index in range(len(systems))
        for method_name in methods
    ]
    generator = np.random.default_rng(seed)
    generator.shuffle(jobs)
    rows: list[dict[str, object]] = []
    gc.collect()
    gc.disable()
    try:
        for repeat, system_index, method_name in jobs:
            system = systems[system_index]
            started = perf_counter_ns()
            _, _, products, accepted = methods[method_name](
                system.matrix, system.gradient, system.plan
            )
            elapsed = perf_counter_ns() - started
            rows.append(
                {
                    "repeat": repeat,
                    "system": system_index,
                    "method": method_name,
                    "products": products,
                    "accepted_degree_two": accepted,
                    "total_ns": elapsed,
                }
            )
    finally:
        gc.enable()
    return rows


def method_statistics(
    timing: list[dict[str, object]], method_name: str
) -> dict[str, float]:
    selected = [row for row in timing if row["method"] == method_name]
    times = np.asarray([float(row["total_ns"]) / 1.0e3 for row in selected])
    products = np.asarray([float(row["products"]) for row in selected])
    return {
        "median_microseconds": float(np.median(times)),
        "mean_microseconds": float(np.mean(times)),
        "p95_microseconds": float(np.percentile(times, 95.0)),
        "median_products": float(np.median(products)),
        "mean_products": float(np.mean(products)),
    }


def configuration_summary(
    systems: list[NewtonSystem],
    qps: list[dict[str, object]],
    validation: list[dict[str, object]],
    timing: list[dict[str, object]],
    method_names: tuple[str, ...],
) -> dict[str, object]:
    by_method = {
        method_name: method_statistics(timing, method_name)
        for method_name in method_names
    }
    lookup = {
        (int(row["repeat"]), int(row["system"]), str(row["method"])): float(
            row["total_ns"]
        )
        for row in timing
    }
    product_lookup = {
        (int(row["repeat"]), int(row["system"]), str(row["method"])): float(
            row["products"]
        )
        for row in timing
    }
    pairs = sorted(
        {
            (int(row["repeat"]), int(row["system"]))
            for row in timing
        }
    )
    comparisons: dict[str, object] = {}
    baseline_name = "interval_chebyshev"
    for method_name in ("certificate", "cost_gated_certificate"):
        comparisons[method_name] = {
            "median_time_ratio_to_chebyshev": (
                by_method[method_name]["median_microseconds"]
                / by_method[baseline_name]["median_microseconds"]
            ),
            "mean_time_ratio_to_chebyshev": (
                by_method[method_name]["mean_microseconds"]
                / by_method[baseline_name]["mean_microseconds"]
            ),
            "paired_strict_win_fraction": float(
                np.mean(
                    [
                        lookup[(repeat, system, method_name)]
                        < lookup[(repeat, system, baseline_name)]
                        for repeat, system in pairs
                    ]
                )
            ),
        }
    # Sum solver time and products over every Newton path before summarizing.
    qp_keys = sorted(
        {
            (system.plant, system.horizon, system.state_index)
            for system in systems
        }
    )
    qp_totals: dict[str, object] = {}
    repeats = sorted({int(row["repeat"]) for row in timing})
    indices_by_qp = {
        key: [
            index
            for index, system in enumerate(systems)
            if (system.plant, system.horizon, system.state_index) == key
        ]
        for key in qp_keys
    }
    for method_name in method_names:
        total_times = []
        total_products = []
        for repeat in repeats:
            for plant, horizon, state_index in qp_keys:
                indices = indices_by_qp[(plant, horizon, state_index)]
                total_times.append(
                    sum(lookup[(repeat, index, method_name)] for index in indices)
                    / 1.0e3
                )
                total_products.append(
                    sum(
                        product_lookup[(repeat, index, method_name)]
                        for index in indices
                    )
                )
        qp_totals[method_name] = {
            "median_total_microseconds": float(np.median(total_times)),
            "mean_total_microseconds": float(np.mean(total_times)),
            "median_total_products": float(np.median(total_products)),
            "mean_total_products": float(np.mean(total_products)),
        }
    certificate_rows = [
        row for row in validation if row["method"] == "certificate"
    ]
    enclosure_inflation = [
        (system.plan.upper / system.plan.lower)
        / (system.exact_upper / system.exact_lower)
        for system in systems
    ]
    return {
        "plant": systems[0].plant,
        "horizon": systems[0].horizon,
        "quadratic_programs": len(qps),
        "reference_convergence_rate": float(
            np.mean([bool(row["reference_converged"]) for row in qps])
        ),
        "newton_systems": len(systems),
        "degree_two_acceptance_fraction": float(
            np.mean(
                [bool(row["accepted_degree_two"]) for row in certificate_rows]
            )
        ),
        "condition_number": {
            "minimum": min(system.plan.upper / system.plan.lower for system in systems),
            "median": float(
                np.median(
                    [system.plan.upper / system.plan.lower for system in systems]
                )
            ),
            "maximum": max(system.plan.upper / system.plan.lower for system in systems),
        },
        "weyl_to_exact_condition_ratio": {
            "median": float(np.median(enclosure_inflation)),
            "maximum": max(enclosure_inflation),
        },
        "chebyshev_degree": {
            "minimum": min(system.plan.degree for system in systems),
            "median": float(np.median([system.plan.degree for system in systems])),
            "maximum": max(system.plan.degree for system in systems),
        },
        "all_residuals_valid": bool(
            all(
                float(row["relative_residual"])
                <= systems[int(row["system"])].plan.tolerance * (1.0 + 2.0e-6)
                for row in validation
                if row["method"] != "direct"
            )
        ),
        "by_method": by_method,
        "comparisons_to_interval_chebyshev": comparisons,
        "per_qp_totals": qp_totals,
    }


def solve_end_to_end_qp(
    *,
    hessian: Array,
    linear_term: Array,
    input_bound: float,
    base_interval: tuple[float, float],
    barrier_weights: tuple[float, ...],
    tolerance: float,
    method_name: str,
    method: Callable[
        [Array, Array, FixedPlan], tuple[Array, Array, int, bool]
    ],
    maximum_iterations: int = 80,
) -> dict[str, object]:
    """Advance one solver along its own damped barrier-Newton trajectory."""

    control = np.zeros_like(linear_term)
    updates = 0
    systems = 0
    products = 0
    acceptances = 0
    backtracks = 0
    converged_stages = 0
    maximum_linear_residual = 0.0
    failure = ""
    for barrier_weight in barrier_weights:
        for _iteration in range(maximum_iterations):
            _, barrier_gradient, barrier_diagonal = barrier_terms(
                control, input_bound, barrier_weight
            )
            gradient = hessian @ control + linear_term + barrier_gradient
            matrix = hessian + np.diag(barrier_diagonal)
            interval = (
                base_interval[0] + float(np.min(barrier_diagonal)),
                base_interval[1] + float(np.max(barrier_diagonal)),
            )
            plan = make_fixed_plan(interval, tolerance)
            direction, _, used_products, accepted = method(
                matrix, gradient, plan
            )
            systems += 1
            products += int(used_products)
            acceptances += int(accepted)
            if method_name != "direct":
                normalizer = float(np.linalg.norm(gradient))
                actual_residual = matrix @ direction + gradient
                relative_residual = float(np.linalg.norm(actual_residual)) / normalizer
                maximum_linear_residual = max(
                    maximum_linear_residual, relative_residual
                )
                if relative_residual > tolerance * (1.0 + 3.0e-6):
                    failure = (
                        f"linear residual {relative_residual:.8g} exceeds target"
                    )
                    break
            directional_derivative = float(gradient @ direction)
            if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
                failure = (
                    f"non-descent direction {directional_derivative:.8g}"
                )
                break
            current_value = barrier_objective(
                control,
                hessian,
                linear_term,
                input_bound,
                barrier_weight,
            )
            decrement_half = -0.5 * directional_derivative
            if decrement_half <= 1.0e-11 * max(1.0, abs(current_value)):
                converged_stages += 1
                break
            step = 0.995 * maximum_interior_step(
                control, direction, input_bound
            )
            local_backtracks = 0
            while barrier_objective(
                control + step * direction,
                hessian,
                linear_term,
                input_bound,
                barrier_weight,
            ) > current_value + 1.0e-4 * step * directional_derivative:
                step *= 0.5
                local_backtracks += 1
                if step < 2.0**-45:
                    failure = "Armijo line search failed"
                    break
            if failure:
                break
            control = control + step * direction
            updates += 1
            backtracks += local_backtracks
        else:
            failure = f"barrier stage exceeded {maximum_iterations} iterations"
        if failure:
            break
    final_weight = barrier_weights[-1]
    _, final_barrier_gradient, _ = barrier_terms(
        control, input_bound, final_weight
    )
    final_gradient = hessian @ control + linear_term + final_barrier_gradient
    quadratic_objective = (
        0.5 * float(control @ hessian @ control)
        + float(linear_term @ control)
    )
    return {
        "method": method_name,
        "converged": bool(
            not failure and converged_stages == len(barrier_weights)
        ),
        "failure": failure,
        "converged_stages": converged_stages,
        "newton_updates": updates,
        "linear_systems": systems,
        "matrix_products": products,
        "degree_two_acceptances": acceptances,
        "line_search_backtracks": backtracks,
        "maximum_linear_relative_residual": maximum_linear_residual,
        "final_scaled_gradient": float(np.linalg.norm(final_gradient))
        / max(1.0, float(np.linalg.norm(linear_term))),
        "quadratic_objective": quadratic_objective,
        "terminal_barrier_objective": barrier_objective(
            control,
            hessian,
            linear_term,
            input_bound,
            final_weight,
        ),
        "maximum_control_fraction": float(
            np.max(np.abs(control)) / input_bound
        ),
        "feasibility_violation": max(
            0.0, float(np.max(np.abs(control)) - input_bound)
        ),
        "solution_json": json.dumps(control.tolist(), separators=(",", ":")),
    }


def end_to_end_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    references = {
        (str(row["plant"]), int(row["horizon"]), int(row["state_index"])): row
        for row in rows
        if row["method"] == "direct"
    }
    method_names = tuple(sorted({str(row["method"]) for row in rows}))
    by_method: dict[str, object] = {}
    for method_name in method_names:
        selected = [row for row in rows if row["method"] == method_name]
        objective_gaps = []
        solution_distances = []
        for row in selected:
            reference = references[
                (
                    str(row["plant"]),
                    int(row["horizon"]),
                    int(row["state_index"]),
                )
            ]
            reference_objective = float(reference["quadratic_objective"])
            objective_gaps.append(
                abs(float(row["quadratic_objective"]) - reference_objective)
                / max(1.0, abs(reference_objective))
            )
            solution = np.asarray(json.loads(str(row["solution_json"])))
            reference_solution = np.asarray(
                json.loads(str(reference["solution_json"]))
            )
            solution_distances.append(
                float(np.linalg.norm(solution - reference_solution))
                / max(1.0, float(np.linalg.norm(reference_solution)))
            )
        total_systems = sum(int(row["linear_systems"]) for row in selected)
        total_acceptances = sum(
            int(row["degree_two_acceptances"]) for row in selected
        )
        by_method[method_name] = {
            "qps": len(selected),
            "converged_qps": sum(bool(row["converged"]) for row in selected),
            "convergence_rate": float(
                np.mean([bool(row["converged"]) for row in selected])
            ),
            "total_newton_updates": sum(
                int(row["newton_updates"]) for row in selected
            ),
            "total_linear_systems": total_systems,
            "total_matrix_products": sum(
                int(row["matrix_products"]) for row in selected
            ),
            "total_line_search_backtracks": sum(
                int(row["line_search_backtracks"]) for row in selected
            ),
            "degree_two_acceptance_rate": (
                total_acceptances / total_systems if total_systems else 0.0
            ),
            "maximum_linear_relative_residual": max(
                float(row["maximum_linear_relative_residual"])
                for row in selected
            ),
            "maximum_final_scaled_gradient": max(
                float(row["final_scaled_gradient"]) for row in selected
            ),
            "maximum_feasibility_violation": max(
                float(row["feasibility_violation"]) for row in selected
            ),
            "fraction_qps_with_control_fraction_above_0_90": float(
                np.mean(
                    [float(row["maximum_control_fraction"]) > 0.90 for row in selected]
                )
            ),
            "fraction_qps_with_control_fraction_above_0_95": float(
                np.mean(
                    [float(row["maximum_control_fraction"]) > 0.95 for row in selected]
                )
            ),
            "maximum_relative_objective_gap_to_direct": max(objective_gaps),
            "median_relative_objective_gap_to_direct": float(
                np.median(objective_gaps)
            ),
            "maximum_relative_solution_distance_to_direct": max(
                solution_distances
            ),
        }
    by_configuration: dict[str, object] = {}
    for plant in sorted({str(row["plant"]) for row in rows}):
        for horizon in sorted({int(row["horizon"]) for row in rows}):
            key = f"{plant}_N{horizon}"
            by_configuration[key] = {}
            for method_name in method_names:
                selected = [
                    row
                    for row in rows
                    if row["plant"] == plant
                    and int(row["horizon"]) == horizon
                    and row["method"] == method_name
                ]
                by_configuration[key][method_name] = {
                    "qps": len(selected),
                    "converged_qps": sum(
                        bool(row["converged"]) for row in selected
                    ),
                    "mean_newton_updates": float(
                        np.mean([int(row["newton_updates"]) for row in selected])
                    ),
                    "mean_matrix_products": float(
                        np.mean([int(row["matrix_products"]) for row in selected])
                    ),
                }
    comparisons = {}
    interval_products = by_method["interval_chebyshev"]["total_matrix_products"]
    for method_name in ("certificate", "cost_gated_certificate", "cg"):
        comparisons[method_name] = {
            "matrix_product_ratio_to_interval_chebyshev": (
                by_method[method_name]["total_matrix_products"]
                / interval_products
            )
        }
    return {
        "by_method": by_method,
        "by_configuration": by_configuration,
        "comparisons_to_interval_chebyshev": comparisons,
    }


def run_end_to_end_validation(
    *,
    plants: list[Plant],
    horizons: list[int],
    states: list[Array],
    barrier_weights: tuple[float, ...],
    tolerance: float,
    gate_degree: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    methods = {
        "certificate": execute_certificate,
        "cost_gated_certificate": cost_gated_solver(gate_degree),
        "interval_chebyshev": execute_interval_chebyshev,
        "cg": execute_cg,
        "direct": execute_direct,
    }
    rows: list[dict[str, object]] = []
    for plant in plants:
        for horizon in horizons:
            hessian, gradient_map = condensed_problem(plant, horizon)
            eigenvalues = np.linalg.eigvalsh(hessian)
            base_interval = (float(eigenvalues[0]), float(eigenvalues[-1]))
            for state_index, state in enumerate(states):
                linear_term = gradient_map @ state
                for method_name, method in methods.items():
                    row = solve_end_to_end_qp(
                        hessian=hessian,
                        linear_term=linear_term,
                        input_bound=plant.input_bound,
                        base_interval=base_interval,
                        barrier_weights=barrier_weights,
                        tolerance=tolerance,
                        method_name=method_name,
                        method=method,
                    )
                    row.update(
                        {
                            "plant": plant.name,
                            "horizon": horizon,
                            "state_index": state_index,
                        }
                    )
                    rows.append(row)
    summary = {
        "protocol": (
            "each method follows its own three-stage barrier-Newton path; "
            "Weyl endpoints use fixed-H extrema and the current barrier "
            "diagonal; no exact current-system eigenvalue is used"
        ),
        "linear_relative_residual_tolerance": tolerance,
        "barrier_weights": barrier_weights,
        "qps_per_method": len(plants) * len(horizons) * len(states),
        **end_to_end_summary(rows),
    }
    return rows, summary


def write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes(project_root: Path) -> dict[str, str]:
    sources = [
        Path(__file__).resolve(),
        *sorted((project_root / "src").rglob("*.py")),
    ]
    test_file = project_root / "experiments" / "test_constrained_mpc.py"
    if test_file.exists():
        sources.append(test_file)
    return {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def write_experiment_summary(
    summary: dict[str, object],
    destination: Path,
    e2e_summary: dict[str, object] | None = None,
) -> None:
    lines = [
        "# Constrained MPC Newton-system experiment",
        "",
        "The paired timing experiment replays time-varying SPD systems collected ",
        "along reference logarithmic-barrier Newton paths. Chebyshev plans are fully prepared ",
        "before timing, and no artificial delay is used.",
        "",
        "| Plant | Horizon | Systems | Acceptance | Median cert/Cheb | Mean cert/Cheb | Median gated/Cheb |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["configurations"]:
        comparisons = result["comparisons_to_interval_chebyshev"]
        lines.append(
            "| {plant} | {horizon} | {systems} | {acceptance:.1%} | {median:.3f} | {mean:.3f} | {gated:.3f} |".format(
                plant=result["plant"],
                horizon=result["horizon"],
                systems=result["newton_systems"],
                acceptance=result["degree_two_acceptance_fraction"],
                median=comparisons["certificate"]["median_time_ratio_to_chebyshev"],
                mean=comparisons["certificate"]["mean_time_ratio_to_chebyshev"],
                gated=comparisons["cost_gated_certificate"]["median_time_ratio_to_chebyshev"],
            )
        )
    lines.extend(
        [
            "",
            "Adaptive CG and direct factorization are reported in `summary.json`; ",
            "the certificate method is not claimed to outperform them. The ",
            "intended comparison is with an unpreconditioned polynomial schedule ",
            "whose coefficients must be fixed before the remaining products are evaluated.",
            "",
        ]
    )
    if e2e_summary is not None:
        by_method = e2e_summary["by_method"]
        certificate = by_method["certificate"]
        chebyshev = by_method["interval_chebyshev"]
        cg = by_method["cg"]
        ratio = (
            certificate["total_matrix_products"]
            / chebyshev["total_matrix_products"]
        )
        lines.extend(
            [
                "## Independent-path barrier solves",
                "",
                "Unlike the paired replay above, this validation advances each method ",
                "along its own barrier-Newton trajectory.",
                "",
                "| Method | Converged QPs | Matrix products |",
                "|---|---:|---:|",
                f"| Certificate | {certificate['converged_qps']}/{certificate['qps']} | {certificate['total_matrix_products']} |",
                f"| Interval Chebyshev | {chebyshev['converged_qps']}/{chebyshev['qps']} | {chebyshev['total_matrix_products']} |",
                f"| CG | {cg['converged_qps']}/{cg['qps']} | {cg['total_matrix_products']} |",
                "",
                f"The certificate/interval-Chebyshev product ratio is {ratio:.3f}. ",
                "The maximum relative quadratic-objective gap to the direct reference is ",
                f"{certificate['maximum_relative_objective_gap_to_direct']:.3e}. ",
                "This is an end-to-end numerical validation of the barrier-QP solves, ",
                "not a closed-loop controller timing comparison.",
                "",
            ]
        )
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plants", default="double_integrator,damped_oscillator")
    parser.add_argument("--horizons", default="20,40,60")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--state-radius", type=float, default=0.75)
    parser.add_argument("--barrier-weights", default="0.1,0.03,0.01")
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--gate-degree", type=int, default=22)
    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="skip the independent-path end-to-end barrier validation",
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timing_repeats < 1 or args.gate_degree < 2:
        raise ValueError("timing repeats and gate degree must be positive")
    selected_names = args.plants.split(",")
    available = {plant.name: plant for plant in benchmark_plants()}
    if any(name not in available for name in selected_names):
        raise ValueError("unknown plant name")
    plants = [available[name] for name in selected_names]
    horizons = [int(value) for value in args.horizons.split(",")]
    barrier_weights = tuple(
        float(value) for value in args.barrier_weights.split(",")
    )
    states = state_panel(
        count=args.states,
        angle_offset=args.angle_offset,
        radius=args.state_radius,
    )
    methods = {
        "certificate": execute_certificate,
        "cost_gated_certificate": cost_gated_solver(args.gate_degree),
        "interval_chebyshev": execute_interval_chebyshev,
        "cg": execute_cg,
        "direct": execute_direct,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    all_systems: list[NewtonSystem] = []
    all_qps: list[dict[str, object]] = []
    all_validation: list[dict[str, object]] = []
    all_timing: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    offset = 0
    for plant in plants:
        for horizon in horizons:
            hessian, gradient_map = condensed_problem(plant, horizon)
            base_eigenvalues = np.linalg.eigvalsh(hessian)
            base_interval = (
                float(base_eigenvalues[0]),
                float(base_eigenvalues[-1]),
            )
            configuration_systems: list[NewtonSystem] = []
            configuration_qps: list[dict[str, object]] = []
            for state_index, state in enumerate(states):
                systems, qp = collect_reference_systems(
                    plant=plant,
                    horizon=horizon,
                    state_index=state_index,
                    state=state,
                    barrier_weights=barrier_weights,
                    tolerance=args.tolerance,
                    precomputed_hessian=hessian,
                    precomputed_gradient_map=gradient_map,
                    precomputed_base_interval=base_interval,
                )
                configuration_systems.extend(systems)
                configuration_qps.append(qp)
            validation = validate_systems(configuration_systems, methods)
            timing = time_systems(
                configuration_systems,
                methods,
                args.timing_repeats,
                args.seed + 1009 * horizon + len(plant.name),
            )
            summaries.append(
                configuration_summary(
                    configuration_systems,
                    configuration_qps,
                    validation,
                    timing,
                    tuple(methods),
                )
            )
            all_systems.extend(configuration_systems)
            all_qps.extend(configuration_qps)
            all_validation.extend(
                [
                    {**row, "system": int(row["system"]) + offset}
                    for row in validation
                ]
            )
            all_timing.extend(
                [
                    {**row, "system": int(row["system"]) + offset}
                    for row in timing
                ]
            )
            offset += len(configuration_systems)
            print(
                f"completed {plant.name} horizon={horizon}: "
                f"{len(configuration_systems)} systems",
                flush=True,
            )
    system_rows = [
        {
            "system": index,
            "plant": system.plant,
            "horizon": system.horizon,
            "state_index": system.state_index,
            "barrier_stage": system.barrier_stage,
            "newton_iteration": system.newton_iteration,
            "barrier_weight": system.barrier_weight,
            "weyl_lower": system.plan.lower,
            "weyl_upper": system.plan.upper,
            "exact_lower": system.exact_lower,
            "exact_upper": system.exact_upper,
            "weyl_condition_number": system.plan.upper / system.plan.lower,
            "exact_condition_number": system.exact_upper / system.exact_lower,
            "interval_chebyshev_degree": system.plan.degree,
            "maximum_control_fraction": system.maximum_control_fraction,
            "newton_decrement_half": system.newton_decrement_half,
        }
        for index, system in enumerate(all_systems)
    ]
    write_csv(system_rows, args.output / "systems.csv")
    write_csv(all_qps, args.output / "qps.csv")
    write_csv(all_validation, args.output / "validation.csv")
    write_csv(all_timing, args.output / "timing.csv")
    project_root = Path(__file__).resolve().parents[1]
    summary = {
        "status": "complete predeclared grid; no configuration or state omitted",
        "experiment_scope": (
            "reference box-barrier Newton-system replay; not an end-to-end "
            "comparison of constrained MPC controllers"
        ),
        "configuration": {
            "plants": selected_names,
            "horizons": horizons,
            "states": args.states,
            "state_angles_radians": [
                args.angle_offset + float(value)
                for value in np.linspace(0.0, np.pi, args.states, endpoint=False)
            ],
            "state_scales": [
                float(np.linalg.norm(state)) for state in states
            ],
            "barrier_weights": barrier_weights,
            "linear_residual_tolerance": args.tolerance,
            "timing_repeats": args.timing_repeats,
            "gate_degree": args.gate_degree,
            "seed": args.seed,
        },
        "endpoint_protocol": (
            "lambda_min/max of the fixed condensed Hessian are computed once; "
            "each Newton system uses the Weyl enclosure lambda_min(H)+min(d) "
            "and lambda_max(H)+max(d); exact endpoints are validation only"
        ),
        "timing_protocol": (
            "single process; solver and system order randomized; no inserted "
            "delay; plant condensation, reference paths, endpoint validation, "
            "and every interval-Chebyshev plan are outside timed regions"
        ),
        "cost_gate_provenance": (
            "degree 22 is a secondary sensitivity fixed from the first "
            "independent dense-SPD timing pilot; its hardware-sensitive value "
            "does not define the ungated primary result"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "configurations": summaries,
        "source_sha256": source_hashes(project_root),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    e2e_summary = None
    if not args.skip_e2e:
        e2e_rows, e2e_summary = run_end_to_end_validation(
            plants=plants,
            horizons=horizons,
            states=states,
            barrier_weights=barrier_weights,
            tolerance=args.tolerance,
            gate_degree=args.gate_degree,
        )
        write_csv(e2e_rows, args.output / "e2e_qps.csv")
        (args.output / "e2e_summary.json").write_text(
            json.dumps(e2e_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_experiment_summary(
        summary,
        args.output / "EXPERIMENT_SUMMARY.md",
        e2e_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
