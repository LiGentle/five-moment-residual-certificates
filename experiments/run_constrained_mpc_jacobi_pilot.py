"""Exploratory fixed-Jacobi robustness check for constrained MPC.

This runner is deliberately separate from ``run_constrained_mpc.py`` and its
frozen ``constrained_mpc_v1`` output.  For a fixed condensed Hessian H, it
uses the symmetric preconditioner P = diag(H)^(-1/2) and solves

    (P A P) y = -P g,  direction = P y,

where A = H + diag(d) is the current log-barrier Newton matrix.  The extrema
of P H P are computed once per plant/horizon.  Weyl's inequalities then give
the online enclosure

    [lambda_min(PHP) + min(P^2 d),
     lambda_max(PHP) + max(P^2 d)].

All polynomial plans are prepared before replay timing.  The declared linear
criterion is the preconditioned relative residual

    ||P(A direction + g)||_2 / ||P g||_2 <= tau.

The ordinary relative residual ||A direction + g||_2 / ||g||_2 is recorded
separately and is not claimed to inherit the same tolerance.  A degree-two
certificate candidate must satisfy both tests; otherwise this conservative
pilot recomputes the complete interval-Chebyshev solve and charges the two
discarded probes.  The fallback and CG still use the declared preconditioned
criterion.  This is an exploratory robustness check, not a superiority claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Callable

import numpy as np

import run_constrained_mpc as base


Array = np.ndarray
Solver = Callable[[Array, Array, base.FixedPlan], tuple[Array, Array, int, bool]]


def original_residual_safeguarded_certificate(scale: Array) -> Solver:
    """Return a certificate solver with a no-product raw-residual check."""

    inverse_scale = 1.0 / scale

    def solve(
        matrix: Array,
        transformed_gradient: Array,
        plan: base.FixedPlan,
    ) -> tuple[Array, Array, int, bool]:
        point, residual, products, accepted = base.execute_certificate(
            matrix, transformed_gradient, plan
        )
        if not accepted:
            return point, residual, products, accepted
        raw_ratio = float(np.linalg.norm(inverse_scale * residual)) / float(
            np.linalg.norm(inverse_scale * transformed_gradient)
        )
        if raw_ratio <= plan.tolerance * (1.0 + 3.0e-6):
            return point, residual, products, accepted
        fallback_point, fallback_residual, fallback_products, _ = (
            base.execute_interval_chebyshev(matrix, transformed_gradient, plan)
        )
        # This pilot intentionally recomputes the two probes.  Charging them is
        # conservative relative to an implementation that reuses those vectors.
        return (
            fallback_point,
            fallback_residual,
            fallback_products + 2,
            False,
        )

    return solve


def method_panel(scale: Array, gate_degree: int) -> dict[str, Solver]:
    certificate = original_residual_safeguarded_certificate(scale)

    def gated(
        matrix: Array,
        gradient: Array,
        plan: base.FixedPlan,
    ) -> tuple[Array, Array, int, bool]:
        if plan.degree < gate_degree:
            return base.execute_interval_chebyshev(matrix, gradient, plan)
        return certificate(matrix, gradient, plan)

    return {
        "certificate": certificate,
        "cost_gated_certificate": gated,
        "interval_chebyshev": base.execute_interval_chebyshev,
        "cg": base.execute_cg,
        "direct": base.execute_direct,
    }


def transform_configuration(
    *,
    plant: base.Plant,
    horizon: int,
    states: list[Array],
    barrier_weights: tuple[float, ...],
    tolerance: float,
) -> tuple[
    Array,
    Array,
    Array,
    tuple[float, float],
    list[base.NewtonSystem],
    list[base.NewtonSystem],
    list[dict[str, object]],
]:
    """Collect a reference panel, then apply one fixed Jacobi transform."""

    hessian, gradient_map = base.condensed_problem(plant, horizon)
    scale = 1.0 / np.sqrt(np.diag(hessian))
    transformed_base = scale[:, None] * hessian * scale[None, :]
    transformed_base_eigenvalues = np.linalg.eigvalsh(transformed_base)
    transformed_base_interval = (
        float(transformed_base_eigenvalues[0]),
        float(transformed_base_eigenvalues[-1]),
    )
    raw_base_eigenvalues = np.linalg.eigvalsh(hessian)
    raw_base_interval = (
        float(raw_base_eigenvalues[0]),
        float(raw_base_eigenvalues[-1]),
    )
    transformed_systems: list[base.NewtonSystem] = []
    raw_systems: list[base.NewtonSystem] = []
    qps: list[dict[str, object]] = []
    for state_index, state in enumerate(states):
        collected, qp = base.collect_reference_systems(
            plant=plant,
            horizon=horizon,
            state_index=state_index,
            state=state,
            barrier_weights=barrier_weights,
            tolerance=tolerance,
            precomputed_hessian=hessian,
            precomputed_gradient_map=gradient_map,
            precomputed_base_interval=raw_base_interval,
        )
        qps.append(qp)
        for system in collected:
            barrier_diagonal = np.diag(system.matrix) - np.diag(hessian)
            transformed_diagonal = scale * scale * barrier_diagonal
            transformed_matrix = scale[:, None] * system.matrix * scale[None, :]
            interval = (
                transformed_base_interval[0] + float(np.min(transformed_diagonal)),
                transformed_base_interval[1] + float(np.max(transformed_diagonal)),
            )
            exact_eigenvalues = np.linalg.eigvalsh(transformed_matrix)
            transformed_systems.append(
                base.NewtonSystem(
                    plant=system.plant,
                    horizon=system.horizon,
                    state_index=system.state_index,
                    barrier_stage=system.barrier_stage,
                    newton_iteration=system.newton_iteration,
                    barrier_weight=system.barrier_weight,
                    matrix=transformed_matrix,
                    gradient=scale * system.gradient,
                    plan=base.make_fixed_plan(interval, tolerance),
                    exact_lower=float(exact_eigenvalues[0]),
                    exact_upper=float(exact_eigenvalues[-1]),
                    maximum_control_fraction=(system.maximum_control_fraction),
                    newton_decrement_half=system.newton_decrement_half,
                )
            )
            raw_systems.append(system)
    return (
        hessian,
        gradient_map,
        scale,
        transformed_base_interval,
        transformed_systems,
        raw_systems,
        qps,
    )


def original_residual_rows(
    transformed_systems: list[base.NewtonSystem],
    raw_systems: list[base.NewtonSystem],
    scale: Array,
    methods: dict[str, Solver],
) -> list[dict[str, object]]:
    """Evaluate both residual definitions outside the timed replay."""

    rows: list[dict[str, object]] = []
    for index, (transformed, raw) in enumerate(
        zip(transformed_systems, raw_systems, strict=True)
    ):
        for method_name, method in methods.items():
            point, transformed_residual, products, accepted = method(
                transformed.matrix,
                transformed.gradient,
                transformed.plan,
            )
            raw_direction = scale * point
            raw_residual = raw.matrix @ raw_direction + raw.gradient
            rows.append(
                {
                    "system": index,
                    "method": method_name,
                    "products": products,
                    "accepted_degree_two": accepted,
                    "preconditioned_relative_residual": (
                        float(np.linalg.norm(transformed_residual))
                        / float(np.linalg.norm(transformed.gradient))
                    ),
                    "original_relative_residual": (
                        float(np.linalg.norm(raw_residual))
                        / float(np.linalg.norm(raw.gradient))
                    ),
                    "original_residual_norm": float(np.linalg.norm(raw_residual)),
                    "original_gradient_norm": float(np.linalg.norm(raw.gradient)),
                }
            )
    return rows


def solve_end_to_end_qp(
    *,
    hessian: Array,
    linear_term: Array,
    input_bound: float,
    transformed_base: Array,
    transformed_base_interval: tuple[float, float],
    scale: Array,
    barrier_weights: tuple[float, ...],
    tolerance: float,
    method_name: str,
    method: Solver,
    maximum_iterations: int = 80,
) -> dict[str, object]:
    """Run one barrier solve using the declared preconditioned criterion."""

    control = np.zeros_like(linear_term)
    updates = 0
    systems = 0
    products = 0
    acceptances = 0
    backtracks = 0
    converged_stages = 0
    maximum_preconditioned_residual = 0.0
    maximum_original_residual = 0.0
    failure = ""
    for barrier_weight in barrier_weights:
        for _iteration in range(maximum_iterations):
            _, barrier_gradient, barrier_diagonal = base.barrier_terms(
                control, input_bound, barrier_weight
            )
            gradient = hessian @ control + linear_term + barrier_gradient
            raw_matrix = hessian + np.diag(barrier_diagonal)
            transformed_diagonal = scale * scale * barrier_diagonal
            transformed_matrix = transformed_base + np.diag(transformed_diagonal)
            transformed_gradient = scale * gradient
            plan = base.make_fixed_plan(
                (
                    transformed_base_interval[0] + float(np.min(transformed_diagonal)),
                    transformed_base_interval[1] + float(np.max(transformed_diagonal)),
                ),
                tolerance,
            )
            transformed_direction, residual, used, accepted = method(
                transformed_matrix,
                transformed_gradient,
                plan,
            )
            direction = scale * transformed_direction
            systems += 1
            products += int(used)
            acceptances += int(accepted)
            if method_name != "direct":
                preconditioned_ratio = float(np.linalg.norm(residual)) / float(
                    np.linalg.norm(transformed_gradient)
                )
                raw_residual = raw_matrix @ direction + gradient
                original_ratio = float(np.linalg.norm(raw_residual)) / float(
                    np.linalg.norm(gradient)
                )
                maximum_preconditioned_residual = max(
                    maximum_preconditioned_residual,
                    preconditioned_ratio,
                )
                maximum_original_residual = max(
                    maximum_original_residual,
                    original_ratio,
                )
                if preconditioned_ratio > tolerance * (1.0 + 3.0e-6):
                    failure = (
                        "preconditioned residual exceeds tolerance: "
                        f"{preconditioned_ratio:.8g}"
                    )
                    break
            directional_derivative = float(gradient @ direction)
            if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
                failure = f"non-descent direction: {directional_derivative:.8g}"
                break
            current_value = base.barrier_objective(
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
            step = 0.995 * base.maximum_interior_step(control, direction, input_bound)
            local_backtracks = 0
            while (
                base.barrier_objective(
                    control + step * direction,
                    hessian,
                    linear_term,
                    input_bound,
                    barrier_weight,
                )
                > current_value + 1.0e-4 * step * directional_derivative
            ):
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
    _, final_barrier_gradient, _ = base.barrier_terms(
        control, input_bound, final_weight
    )
    final_gradient = hessian @ control + linear_term + final_barrier_gradient
    objective = 0.5 * float(control @ hessian @ control) + float(linear_term @ control)
    return {
        "method": method_name,
        "converged": bool(not failure and converged_stages == len(barrier_weights)),
        "failure": failure,
        "newton_updates": updates,
        "linear_systems": systems,
        "matrix_products": products,
        "degree_two_acceptances": acceptances,
        "line_search_backtracks": backtracks,
        "maximum_preconditioned_relative_residual": (maximum_preconditioned_residual),
        "maximum_original_relative_residual": maximum_original_residual,
        "final_scaled_gradient": float(np.linalg.norm(final_gradient))
        / max(1.0, float(np.linalg.norm(linear_term))),
        "quadratic_objective": objective,
        "maximum_control_fraction": float(np.max(np.abs(control)) / input_bound),
        "feasibility_violation": max(0.0, float(np.max(np.abs(control)) - input_bound)),
        "solution_json": json.dumps(control.tolist(), separators=(",", ":")),
    }


def add_end_to_end_comparisons(
    rows: list[dict[str, object]],
) -> None:
    references = {
        (str(row["plant"]), int(row["horizon"]), int(row["state_index"])): row
        for row in rows
        if row["method"] == "direct"
    }
    for row in rows:
        reference = references[
            (
                str(row["plant"]),
                int(row["horizon"]),
                int(row["state_index"]),
            )
        ]
        reference_objective = float(reference["quadratic_objective"])
        row["relative_objective_gap_to_direct"] = abs(
            float(row["quadratic_objective"]) - reference_objective
        ) / max(1.0, abs(reference_objective))
        solution = np.asarray(json.loads(str(row["solution_json"])))
        reference_solution = np.asarray(json.loads(str(reference["solution_json"])))
        row["relative_solution_distance_to_direct"] = float(
            np.linalg.norm(solution - reference_solution)
        ) / max(1.0, float(np.linalg.norm(reference_solution)))


def end_to_end_summary(
    rows: list[dict[str, object]],
    method_names: tuple[str, ...],
    plants: list[base.Plant],
    horizons: list[int],
) -> dict[str, object]:
    by_method: dict[str, object] = {}
    for method_name in method_names:
        selected = [row for row in rows if row["method"] == method_name]
        total_systems = sum(int(row["linear_systems"]) for row in selected)
        total_acceptances = sum(int(row["degree_two_acceptances"]) for row in selected)
        by_method[method_name] = {
            "qps": len(selected),
            "converged_qps": sum(bool(row["converged"]) for row in selected),
            "total_newton_updates": sum(int(row["newton_updates"]) for row in selected),
            "total_linear_systems": total_systems,
            "total_matrix_products": sum(
                int(row["matrix_products"]) for row in selected
            ),
            "degree_two_acceptance_rate": (
                total_acceptances / total_systems if total_systems else 0.0
            ),
            "maximum_preconditioned_relative_residual": max(
                float(row["maximum_preconditioned_relative_residual"])
                for row in selected
            ),
            "maximum_original_relative_residual": max(
                float(row["maximum_original_relative_residual"]) for row in selected
            ),
            "maximum_feasibility_violation": max(
                float(row["feasibility_violation"]) for row in selected
            ),
            "maximum_relative_objective_gap_to_direct": max(
                float(row["relative_objective_gap_to_direct"]) for row in selected
            ),
            "maximum_relative_solution_distance_to_direct": max(
                float(row["relative_solution_distance_to_direct"]) for row in selected
            ),
        }
    by_configuration: dict[str, object] = {}
    for plant in plants:
        for horizon in horizons:
            key = f"{plant.name}_N{horizon}"
            by_configuration[key] = {}
            for method_name in method_names:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method_name
                    and row["plant"] == plant.name
                    and int(row["horizon"]) == horizon
                ]
                by_configuration[key][method_name] = {
                    "qps": len(selected),
                    "converged_qps": sum(bool(row["converged"]) for row in selected),
                    "total_matrix_products": sum(
                        int(row["matrix_products"]) for row in selected
                    ),
                    "maximum_original_relative_residual": max(
                        float(row["maximum_original_relative_residual"])
                        for row in selected
                    ),
                    "maximum_relative_objective_gap_to_direct": max(
                        float(row["relative_objective_gap_to_direct"])
                        for row in selected
                    ),
                    "maximum_feasibility_violation": max(
                        float(row["feasibility_violation"]) for row in selected
                    ),
                }
    baseline_products = by_method["interval_chebyshev"]["total_matrix_products"]
    return {
        "by_method": by_method,
        "by_configuration": by_configuration,
        "certificate_product_ratio_to_interval_chebyshev": (
            by_method["certificate"]["total_matrix_products"] / baseline_products
        ),
    }


def write_csv(rows: list[dict[str, object]], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes(project_root: Path) -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        project_root / "experiments" / "run_constrained_mpc.py",
        project_root / "experiments" / "test_constrained_mpc_jacobi.py",
        *sorted((project_root / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(project_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
        if path.exists()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="20,40,60")
    parser.add_argument("--barrier-weights", default="0.1,0.03,0.01")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--gate-degree", type=int, default=22)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/constrained_mpc_jacobi_pilot"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plants = base.benchmark_plants()
    horizons = [int(value) for value in args.horizons.split(",")]
    barrier_weights = tuple(float(value) for value in args.barrier_weights.split(","))
    states = base.state_panel()
    method_names = (
        "certificate",
        "cost_gated_certificate",
        "interval_chebyshev",
        "cg",
        "direct",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    configuration_summaries: list[dict[str, object]] = []
    all_system_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []
    all_original_residual_rows: list[dict[str, object]] = []
    all_timing_rows: list[dict[str, object]] = []
    all_e2e_rows: list[dict[str, object]] = []
    system_offset = 0

    for plant in plants:
        for horizon in horizons:
            (
                hessian,
                gradient_map,
                scale,
                transformed_base_interval,
                transformed_systems,
                raw_systems,
                qps,
            ) = transform_configuration(
                plant=plant,
                horizon=horizon,
                states=states,
                barrier_weights=barrier_weights,
                tolerance=args.tolerance,
            )
            methods = method_panel(scale, args.gate_degree)
            validation = base.validate_systems(transformed_systems, methods)
            original_rows = original_residual_rows(
                transformed_systems,
                raw_systems,
                scale,
                methods,
            )
            timing = base.time_systems(
                transformed_systems,
                methods,
                args.timing_repeats,
                args.seed + 1009 * horizon + len(plant.name),
            )
            configuration = base.configuration_summary(
                transformed_systems,
                qps,
                validation,
                timing,
                method_names,
            )
            configuration["original_residual_diagnostics"] = {}
            for method_name in method_names:
                selected = [
                    row for row in original_rows if row["method"] == method_name
                ]
                configuration["original_residual_diagnostics"][method_name] = {
                    "maximum": max(
                        float(row["original_relative_residual"]) for row in selected
                    ),
                    "fraction_at_most_tolerance": float(
                        np.mean(
                            [
                                float(row["original_relative_residual"])
                                <= args.tolerance * (1.0 + 3.0e-6)
                                for row in selected
                            ]
                        )
                    ),
                }
            configuration_summaries.append(configuration)

            for index, (transformed, raw) in enumerate(
                zip(transformed_systems, raw_systems, strict=True)
            ):
                all_system_rows.append(
                    {
                        "system": system_offset + index,
                        "plant": transformed.plant,
                        "horizon": transformed.horizon,
                        "state_index": transformed.state_index,
                        "barrier_stage": transformed.barrier_stage,
                        "newton_iteration": transformed.newton_iteration,
                        "barrier_weight": transformed.barrier_weight,
                        "preconditioned_weyl_lower": transformed.plan.lower,
                        "preconditioned_weyl_upper": transformed.plan.upper,
                        "preconditioned_exact_lower": transformed.exact_lower,
                        "preconditioned_exact_upper": transformed.exact_upper,
                        "interval_chebyshev_degree": transformed.plan.degree,
                        "raw_gradient_norm": float(np.linalg.norm(raw.gradient)),
                        "preconditioned_gradient_norm": float(
                            np.linalg.norm(transformed.gradient)
                        ),
                    }
                )
            all_validation_rows.extend(
                {
                    **row,
                    "system": int(row["system"]) + system_offset,
                }
                for row in validation
            )
            all_original_residual_rows.extend(
                {
                    **row,
                    "system": int(row["system"]) + system_offset,
                }
                for row in original_rows
            )
            all_timing_rows.extend(
                {
                    **row,
                    "system": int(row["system"]) + system_offset,
                }
                for row in timing
            )
            system_offset += len(transformed_systems)

            transformed_base = scale[:, None] * hessian * scale[None, :]
            for state_index, state in enumerate(states):
                local_rows: list[dict[str, object]] = []
                for method_name, method in methods.items():
                    row = solve_end_to_end_qp(
                        hessian=hessian,
                        linear_term=gradient_map @ state,
                        input_bound=plant.input_bound,
                        transformed_base=transformed_base,
                        transformed_base_interval=(transformed_base_interval),
                        scale=scale,
                        barrier_weights=barrier_weights,
                        tolerance=args.tolerance,
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
                    local_rows.append(row)
                all_e2e_rows.extend(local_rows)
            print(
                f"completed {plant.name} horizon={horizon}: "
                f"{len(transformed_systems)} systems",
                flush=True,
            )

    add_end_to_end_comparisons(all_e2e_rows)
    e2e = end_to_end_summary(
        all_e2e_rows,
        method_names,
        plants,
        horizons,
    )
    project_root = Path(__file__).resolve().parents[1]
    protocol = {
        "status": "exploratory robustness check; not a superiority claim",
        "preconditioner": (
            "fixed symmetric Jacobi P=diag(H)^(-1/2), computed once per "
            "plant/horizon configuration"
        ),
        "transformed_system": "(P A P) y = -P g; direction = P y",
        "preconditioned_residual_definition": (
            "norm(P(A direction + g),2) / norm(P g,2)"
        ),
        "original_residual_definition": ("norm(A direction + g,2) / norm(g,2)"),
        "declared_stopping_criterion": "preconditioned residual <= 0.05",
        "original_residual_scope": (
            "diagnostic only; it is not claimed to inherit the 0.05 bound"
        ),
        "endpoint_rule": (
            "precompute extrema of P H P; for each barrier diagonal d use "
            "Weyl bounds lambda_min(PHP)+min(P^2 d) and "
            "lambda_max(PHP)+max(P^2 d); exact current eigenvalues are "
            "validation only"
        ),
        "certificate_safeguard": (
            "a degree-two candidate must also meet the original residual "
            "target; otherwise a full interval-Chebyshev solve is recomputed "
            "and the two discarded products are charged"
        ),
        "timing": (
            "single process; five randomized interleaved repeats; transformed "
            "matrices and complete polynomial plans prepared before timing; "
            "no artificial delay; explicit dense transformed products omit "
            "the common O(n) diagonal scaling cost"
        ),
        "configuration": {
            "plants": [plant.name for plant in plants],
            "horizons": horizons,
            "states": len(states),
            "state_radius": 0.75,
            "state_angles": [
                float(value)
                for value in np.linspace(0.0, np.pi, len(states), endpoint=False)
            ],
            "barrier_weights": barrier_weights,
            "tolerance": args.tolerance,
            "timing_repeats": args.timing_repeats,
            "gate_degree": args.gate_degree,
            "seed": args.seed,
        },
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
        "source_sha256": source_hashes(project_root),
    }
    summary = {
        "protocol": protocol,
        "configurations": configuration_summaries,
        "end_to_end": e2e,
    }
    write_csv(all_system_rows, args.output / "systems.csv")
    write_csv(all_validation_rows, args.output / "validation.csv")
    write_csv(
        all_original_residual_rows,
        args.output / "original_residuals.csv",
    )
    write_csv(all_timing_rows, args.output / "timing.csv")
    write_csv(all_e2e_rows, args.output / "e2e_qps.csv")
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
