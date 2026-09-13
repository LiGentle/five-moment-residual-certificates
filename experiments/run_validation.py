"""Held-out validation for the five-moment certified polynomial method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import cvxpy as cp
from numpy.polynomial.chebyshev import chebmul, chebval

from fmcert import (
    chebyshev_budget_for_tolerance,
    chebyshev_polynomial_at_zero,
    correction_chebyshev_coefficients,
    design_five_moment_polynomial,
    design_moment_polynomial,
    evaluate_with_cached_products,
    probe_five_moment_state,
    run_bb,
    run_cg,
    run_minres,
)

Array = np.ndarray


@dataclass(frozen=True)
class SpectralCase:
    name: str
    family: str
    source: str
    eigenvalues: Array
    residual: Array
    interval: tuple[float, float]


def _initial_residual(
    rng: np.random.Generator, eigenvalues: Array, replicate: int
) -> Array:
    vector = rng.normal(size=len(eigenvalues))
    return eigenvalues * vector if replicate % 2 == 0 else vector


def _synthetic_spectrum(
    rng: np.random.Generator, family: str, size: int, condition: float
) -> Array:
    if family == "two_point":
        values = np.r_[np.ones(size // 2), np.full(size - size // 2, condition)]
    elif family == "three_point":
        values = np.r_[
            np.ones(size // 2),
            np.full(size // 4, np.sqrt(condition)),
            np.full(size - 3 * size // 4, condition),
        ]
    elif family == "three_cluster":
        centers = (1.0, np.sqrt(condition), condition)
        counts = (size // 2, size // 4, size - 3 * size // 4)
        values = np.concatenate(
            [
                center * np.exp(rng.uniform(-0.012, 0.012, count))
                for center, count in zip(centers, counts)
            ]
        )
        values = np.clip(values, 1.0, condition)
    elif family == "log_uniform":
        values = np.exp(rng.uniform(0.0, np.log(condition), size))
    elif family == "single_outlier":
        values = np.r_[
            np.exp(rng.uniform(0.0, np.log(min(6.0, condition)), size - 1)),
            condition,
        ]
    else:
        raise ValueError(f"unknown family {family}")
    values[[0, -1]] = [1.0, condition]
    return np.asarray(values, dtype=float)


def build_cases(seed: int, size: int) -> list[SpectralCase]:
    rng = np.random.default_rng(seed)
    cases: list[SpectralCase] = []
    for family in (
        "two_point",
        "three_point",
        "three_cluster",
        "log_uniform",
        "single_outlier",
    ):
        for condition in (10.0, 100.0, 1000.0):
            for replicate in range(4):
                spectrum = _synthetic_spectrum(rng, family, size, condition)
                cases.append(
                    SpectralCase(
                        f"{family}_k{int(condition)}_{replicate}",
                        family,
                        "heldout_synthetic",
                        spectrum,
                        _initial_residual(rng, spectrum, replicate),
                        (1.0, condition),
                    )
                )

    for dimension in (64, 128, 256):
        indices = np.arange(1, dimension + 1)
        spectrum = 2.0 - 2.0 * np.cos(indices * np.pi / (dimension + 1.0))
        for replicate in range(4):
            cases.append(
                SpectralCase(
                    f"poisson1d_n{dimension}_{replicate}",
                    "poisson1d",
                    "heldout_structured",
                    spectrum,
                    _initial_residual(rng, spectrum, replicate),
                    (float(spectrum.min()), float(spectrum.max())),
                )
            )
    for side in (8, 12, 16):
        indices = np.arange(1, side + 1)
        one = 2.0 - 2.0 * np.cos(indices * np.pi / (side + 1.0))
        spectrum = (one[:, None] + one[None, :]).ravel()
        for replicate in range(4):
            cases.append(
                SpectralCase(
                    f"poisson2d_n{side * side}_{replicate}",
                    "poisson2d",
                    "heldout_structured",
                    spectrum,
                    _initial_residual(rng, spectrum, replicate),
                    (float(spectrum.min()), float(spectrum.max())),
                )
            )

    from sklearn.datasets import load_breast_cancer, load_wine

    datasets = {
        "ridge_breast_cancer": load_breast_cancer().data,
        "ridge_wine": load_wine().data,
    }
    for family, raw in datasets.items():
        features = np.asarray(raw, dtype=float)
        features -= features.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(features, axis=0)
        features = features[:, norms > 1.0e-12] / norms[norms > 1.0e-12]
        gram_spectrum = np.linalg.eigvalsh(features.T @ features)
        maximum = float(gram_spectrum[-1])
        for condition in (30.0, 100.0, 1000.0):
            ridge = maximum / (condition - 1.0)
            spectrum = gram_spectrum + ridge
            interval = (ridge, ridge + maximum)
            for replicate in range(4):
                cases.append(
                    SpectralCase(
                        f"{family}_k{int(condition)}_{replicate}",
                        family,
                        "heldout_real_data",
                        spectrum,
                        _initial_residual(rng, spectrum, replicate),
                        interval,
                    )
                )
    return cases


def _actual_squared(case: SpectralCase, coefficients: Array) -> float:
    lower, upper = case.interval
    z = (2.0 * case.eigenvalues - upper - lower) / (upper - lower)
    values = chebval(z, coefficients)
    return float(np.sum(case.residual**2 * values**2) / (case.residual @ case.residual))


def _coverage_slack(actual: float, certificate: float, coefficients: Array) -> float:
    """Roundoff-scale validation slack, not part of the mathematical bound."""

    squared = chebmul(np.asarray(coefficients, dtype=float), coefficients)
    scale = 1.0 + abs(float(actual)) + abs(float(certificate)) + float(
        np.sum(np.abs(squared))
    )
    return 512.0 * np.finfo(float).eps * scale


def _designs(
    case: SpectralCase, degrees: tuple[int, ...]
) -> tuple[Array, dict[int, object], object]:
    state = probe_five_moment_state(
        lambda vector: case.eigenvalues * vector,
        case.residual,
        case.interval,
    )
    eta = state.moments
    designs: dict[int, object] = {}
    for degree in degrees:
        try:
            designs[degree] = design_five_moment_polynomial(
                eta, degree, case.interval
            )
        except RuntimeError:
            continue
    return eta, designs, state


def _baseline_runs(case: SpectralCase, tolerance: float, budget: int) -> dict[str, object]:
    return {
        "cg": run_cg(case.eigenvalues, case.residual, tolerance=tolerance, max_products=budget),
        "minres": run_minres(
            case.eigenvalues, case.residual, tolerance=tolerance, max_products=budget
        ),
        "bb1": run_bb(
            case.eigenvalues,
            case.residual,
            rule="bb1",
            tolerance=tolerance,
            max_products=budget,
        ),
        "bb2": run_bb(
            case.eigenvalues,
            case.residual,
            rule="bb2",
            tolerance=tolerance,
            max_products=budget,
        ),
    }


def evaluate(
    cases: list[SpectralCase],
    degrees: tuple[int, ...],
    tolerances: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    target_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases, start=1):
        start = perf_counter()
        eta, designs, state = _designs(case, degrees)
        nominal_design_seconds = perf_counter() - start
        for degree, design in designs.items():
            actual = _actual_squared(case, design.coefficients)
            excess = actual - design.certificate
            coverage_slack = _coverage_slack(
                actual, design.certificate, design.coefficients
            )
            chebyshev = chebyshev_polynomial_at_zero(degree, case.interval)
            chebyshev_actual = _actual_squared(case, chebyshev)
            lower, upper = case.interval
            z0 = -(upper + lower) / (upper - lower)
            interval_bound = 1.0 / float(np.cosh(degree * np.arccosh(-z0))) ** 2
            design_rows.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "source": case.source,
                    "degree": degree,
                    "certificate": design.certificate,
                    "design_value": design.design_value,
                    "actual_squared_residual": actual,
                    "certificate_to_actual": design.certificate / max(actual, 1.0e-300),
                    "chebyshev_interval_bound": interval_bound,
                    "certificate_to_chebyshev_bound": design.certificate / interval_bound,
                    "actual_to_chebyshev_actual": actual / max(chebyshev_actual, 1.0e-300),
                    "raw_certificate_excess": excess,
                    "coverage_slack": coverage_slack,
                    "scaled_certificate_excess": excess / coverage_slack,
                    "covered": excess <= coverage_slack,
                    "atomic_pattern": design.atomic_pattern,
                    "solver": design.solver,
                }
            )

        if case_index <= 36 and 4 in designs:
            nominal = designs[4]
            signs = np.asarray([0.0, 1.0, -1.0, 1.0, -1.0])
            radii = np.asarray([0.0, 2.0e-6, 2.0e-6, 3.0e-6, 3.0e-6])
            estimate = eta + signs * 0.75 * radii
            robust = design_five_moment_polynomial(
                estimate, 4, case.interval, moment_radii=radii
            )
            actual = _actual_squared(case, robust.coefficients)
            excess = actual - robust.certificate
            coverage_slack = _coverage_slack(
                actual, robust.certificate, robust.coefficients
            )
            robustness_rows.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "actual_squared_residual": actual,
                    "nominal_certificate": nominal.certificate,
                    "robust_design_value": robust.design_value,
                    "robust_certificate": robust.certificate,
                    "raw_certificate_excess": excess,
                    "coverage_slack": coverage_slack,
                    "scaled_certificate_excess": excess / coverage_slack,
                    "covered": excess <= coverage_slack,
                    "inflation": robust.certificate - nominal.certificate,
                    "interval_correction": robust.interval_correction,
                    "status": robust.status,
                    "solver": robust.solver,
                    "dual_l1": float(np.sum(np.abs(robust.dual_majorant))),
                }
            )

        for tolerance in tolerances:
            budget = chebyshev_budget_for_tolerance(case.interval, tolerance)
            feasible = [
                design
                for degree, design in sorted(designs.items())
                if degree < budget and design.certificate <= tolerance * tolerance
            ]
            selected = feasible[0] if feasible else None
            if selected is None:
                coefficients = chebyshev_polynomial_at_zero(budget, case.interval)
                products = budget
                certificate = tolerance * tolerance
            else:
                coefficients = selected.coefficients
                products = selected.degree
                certificate = selected.certificate
            spectral_squared = _actual_squared(case, coefficients)
            final_residual, basis = evaluate_with_cached_products(
                lambda vector: case.eigenvalues * vector,
                case.residual,
                state,
                coefficients,
            )
            initial_squared = float(case.residual @ case.residual)
            actual_squared = float(final_residual @ final_residual) / initial_squared
            excess = actual_squared - certificate
            coverage_slack = _coverage_slack(
                actual_squared, certificate, coefficients
            )
            correction = correction_chebyshev_coefficients(
                coefficients, case.interval
            )
            correction_vector = sum(
                float(correction[index]) * basis[index]
                for index in range(len(correction))
            )
            point = -np.asarray(correction_vector, dtype=float)
            point_residual = case.eigenvalues * point + case.residual
            point_discrepancy = (
                float(np.linalg.norm(point_residual - final_residual))
                / float(np.linalg.norm(case.residual))
            )
            baselines = _baseline_runs(case, tolerance, budget)
            row: dict[str, object] = {
                "name": case.name,
                "family": case.family,
                "source": case.source,
                "dimension": len(case.eigenvalues),
                "condition": case.interval[1] / case.interval[0],
                "tolerance": tolerance,
                "chebyshev_budget": budget,
                "certificate_passed": selected is not None,
                "selected_degree": products,
                "hessian_products": products,
                "collective_reductions": 1,
                "hessian_product_saving": 1.0 - products / budget,
                "certified_squared_residual": certificate,
                "actual_relative_residual": np.sqrt(actual_squared),
                "spectral_recurrence_squared_discrepancy": abs(
                    spectral_squared - actual_squared
                ),
                "point_residual_relative_discrepancy": point_discrepancy,
                "raw_certificate_excess": excess,
                "coverage_slack": coverage_slack,
                "scaled_certificate_excess": excess / coverage_slack,
                "covered": excess <= coverage_slack,
                "converged": actual_squared <= tolerance * tolerance * (1.0 + 2.0e-6),
                "design_seconds": nominal_design_seconds,
            }
            for name, run in baselines.items():
                row[f"{name}_products"] = run.hessian_products
                row[f"{name}_reductions"] = run.scalar_reductions
                row[f"{name}_converged"] = run.converged
                row[f"{name}_final_residual"] = run.final_residual_ratio
            target_rows.append(row)
        print(f"[{case_index:03d}/{len(cases):03d}] {case.name}", flush=True)
    return target_rows, design_rows, robustness_rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(
    target_rows: list[dict[str, object]],
    design_rows: list[dict[str, object]],
    robustness_rows: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "independent validation after source freeze",
        "seed": seed,
        "target_runs": len(target_rows),
        "design_checks": len(design_rows),
        "certificate_violations": sum(not bool(row["covered"]) for row in design_rows),
        "target_certificate_violations": sum(not bool(row["covered"]) for row in target_rows),
        "target_failures": sum(not bool(row["converged"]) for row in target_rows),
        "end_to_end_checks": len(target_rows),
        "max_spectral_recurrence_squared_discrepancy": max(
            float(row["spectral_recurrence_squared_discrepancy"])
            for row in target_rows
        ),
        "max_point_residual_relative_discrepancy": max(
            float(row["point_residual_relative_discrepancy"])
            for row in target_rows
        ),
        "robust_checks": len(robustness_rows),
        "robust_violations": sum(not bool(row["covered"]) for row in robustness_rows),
        "max_raw_certificate_excess": max(
            float(row["raw_certificate_excess"]) for row in design_rows
        ),
        "max_scaled_certificate_excess": max(
            float(row["scaled_certificate_excess"]) for row in design_rows
        ),
        "max_target_raw_certificate_excess": max(
            float(row["raw_certificate_excess"]) for row in target_rows
        ),
        "max_target_scaled_certificate_excess": max(
            float(row["scaled_certificate_excess"]) for row in target_rows
        ),
        "max_robust_raw_certificate_excess": max(
            float(row["raw_certificate_excess"]) for row in robustness_rows
        ),
        "max_robust_scaled_certificate_excess": max(
            float(row["scaled_certificate_excess"]) for row in robustness_rows
        ),
    }
    design_time_by_case = {
        str(row["name"]): float(row["design_seconds"]) for row in target_rows
    }
    summary["candidate_grid_design_seconds"] = {
        "median": float(np.median(list(design_time_by_case.values()))),
        "maximum": float(np.max(list(design_time_by_case.values()))),
    }
    robust_ratios = [
        float(row["robust_certificate"])
        / max(float(row["robust_design_value"]), 1.0e-300)
        for row in robustness_rows
    ]
    summary["robust_certificate_to_design"] = {
        "median": _median(robust_ratios),
        "maximum": float(np.max(robust_ratios)) if robust_ratios else None,
    }
    by_tolerance: dict[str, object] = {}
    for tolerance in sorted({float(row["tolerance"]) for row in target_rows}):
        rows = [row for row in target_rows if float(row["tolerance"]) == tolerance]
        active = [row for row in rows if bool(row["certificate_passed"])]
        family_passes = Counter(str(row["family"]) for row in active)
        by_tolerance[f"{tolerance:.0e}"] = {
            "cases": len(rows),
            "pass_rate": len(active) / len(rows),
            "pass_counts_by_family": dict(family_passes),
            "median_hvp_saving_when_passed": _median(
                [float(row["hessian_product_saving"]) for row in active]
            ),
            "strict_hvp_wins_over_cg": sum(
                int(row["hessian_products"]) < int(row["cg_products"]) for row in rows
            ),
            "strict_hvp_wins_over_minres": sum(
                int(row["hessian_products"]) < int(row["minres_products"]) for row in rows
            ),
        }
    summary["by_tolerance"] = by_tolerance
    by_degree: dict[str, object] = {}
    for degree in sorted({int(row["degree"]) for row in design_rows}):
        rows = [row for row in design_rows if int(row["degree"]) == degree]
        nonzero = [row for row in rows if float(row["certificate"]) > 1.0e-14]
        by_degree[str(degree)] = {
            "cases": len(rows),
            "median_certificate_to_chebyshev_bound": _median(
                [float(row["certificate_to_chebyshev_bound"]) for row in rows]
            ),
            "fraction_strictly_below_chebyshev_bound": float(
                np.mean(
                    [float(row["certificate_to_chebyshev_bound"]) < 1.0 - 1.0e-5 for row in rows]
                )
            ),
            "median_certificate_to_actual_nonzero": _median(
                [float(row["certificate_to_actual"]) for row in nonzero]
            ),
        }
    summary["by_degree"] = by_degree
    return summary


def make_figures(
    target_rows: list[dict[str, object]], design_rows: list[dict[str, object]], output: Path
) -> None:
    import matplotlib.pyplot as plt

    degrees = sorted({int(row["degree"]) for row in design_rows})
    data = [
        [
            min(1.2, float(row["certificate_to_chebyshev_bound"]))
            for row in design_rows
            if int(row["degree"]) == degree
        ]
        for degree in degrees
    ]
    figure, axis = plt.subplots(figsize=(6.2, 3.7))
    axis.boxplot(data, tick_labels=[str(value) for value in degrees], showfliers=False)
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_xlabel("polynomial degree")
    axis.set_ylabel("five-moment bound / interval-only bound")
    axis.set_ylim(0.0, 1.2)
    figure.tight_layout()
    figure.savefig(output / "certificate_improvement.pdf")
    figure.savefig(output / "certificate_improvement.png", dpi=180)
    plt.close(figure)

    tolerances = sorted({float(row["tolerance"]) for row in target_rows}, reverse=True)
    family_order = (
        "two_point",
        "three_point",
        "three_cluster",
        "log_uniform",
        "single_outlier",
        "poisson1d",
        "poisson2d",
        "ridge_breast_cancer",
        "ridge_wine",
    )
    families = [family for family in family_order if any(
        str(row["family"]) == family for row in target_rows
    )]
    family_labels = {
        "two_point": "Two-point",
        "three_point": "Three-point",
        "three_cluster": "Three-cluster",
        "log_uniform": "Log-uniform",
        "single_outlier": "Single-outlier",
        "poisson1d": "1D Poisson",
        "poisson2d": "2D Poisson",
        "ridge_breast_cancer": "Breast-cancer ridge",
        "ridge_wine": "Wine ridge",
    }
    matrix = np.zeros((len(families), len(tolerances)))
    for i, family in enumerate(families):
        for j, tolerance in enumerate(tolerances):
            rows = [
                row
                for row in target_rows
                if row["family"] == family and float(row["tolerance"]) == tolerance
            ]
            matrix[i, j] = np.mean([bool(row["certificate_passed"]) for row in rows])
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(tolerances)), [f"{value:.0e}" for value in tolerances])
    axis.set_yticks(range(len(families)), [family_labels[name] for name in families])
    axis.set_xlabel("relative residual tolerance")
    for i in range(len(families)):
        for j in range(len(tolerances)):
            axis.text(
                j,
                i,
                f"{matrix[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axis, label="fraction accepted early")
    figure.tight_layout()
    figure.savefig(output / "target_activation.pdf")
    figure.savefig(output / "target_activation.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--degrees", default="2,3,4,6,8")
    parser.add_argument("--tolerances", default="1e-2,1e-4,1e-6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    degrees = tuple(int(value) for value in arguments.degrees.split(","))
    tolerances = tuple(float(value) for value in arguments.tolerances.split(","))
    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("CLARABEL is required for the frozen validation run")
    cases = build_cases(arguments.seed, arguments.size)
    if len(cases) != 108:
        raise RuntimeError(f"expected 108 benchmark cases, obtained {len(cases)}")
    started = perf_counter()
    target, designs, robustness = evaluate(cases, degrees, tolerances)
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_csv(target, arguments.output / "target_results.csv")
    _write_csv(designs, arguments.output / "design_results.csv")
    _write_csv(robustness, arguments.output / "robustness_results.csv")
    summary = summarize(target, designs, robustness, arguments.seed)
    summary["wall_seconds"] = perf_counter() - started
    project_root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), *sorted((project_root / "src").rglob("*.py"))]
    summary["source_sha256"] = {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    summary["configuration"] = {
        "size": arguments.size,
        "degrees": list(degrees),
        "tolerances": list(tolerances),
    }
    summary["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cvxpy": cp.__version__,
        "clarabel": version("clarabel"),
        "scs": version("scs"),
        "scikit_learn": version("scikit-learn"),
        "matplotlib": version("matplotlib"),
        "installed_solvers": cp.installed_solvers(),
        "platform": platform.platform(),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_figures(target, designs, arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
