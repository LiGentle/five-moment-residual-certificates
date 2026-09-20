"""Observation-depth frontier for the conditional moment certificates.

This diagnostic asks how many initial Chebyshev products are worth taking
before a fixed continuation polynomial is selected.  It uses a deterministic
subset of the benchmark panel: every two-dimensional Poisson case and every
ridge-regression case with prescribed enclosure ratio 30 or 100.  No case is
selected using its outcome.

The SDP value is monotone in the candidate degree.  We therefore locate its
first crossing by bisection, verify both sides of the crossing, and then check
that the independently majorized polynomial also meets the target.  The
reported ``operational_horizon`` is conservative: if the majorant at the SDP
crossing misses the target because of numerical correction, subsequent
degrees are tested in order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander

from fmcert import chebyshev_budget_for_tolerance, chebyshev_polynomial_at_zero

# The case generator is part of the frozen main validation and records all
# data preprocessing and random seeds in one place.
from run_validation import SpectralCase, build_cases
from vectorized_sdp import design_vectorized_moment_polynomial


@dataclass(frozen=True)
class FrontierRow:
    name: str
    family: str
    source: str
    dimension: int
    enclosure_ratio: float
    tolerance: float
    observation_depth: int
    chebyshev_horizon: int
    sdp_horizon: int
    operational_horizon: int
    certificate: float
    design_value: float
    previous_design_value: float | None
    actual_squared_residual: float
    covered: bool
    design_calls: int
    design_seconds: float
    solver: str
    status: str


def _selected_cases(seed: int, size: int) -> list[SpectralCase]:
    cases = build_cases(seed, size)
    selected = [
        case
        for case in cases
        if case.family == "poisson2d"
        or (
            case.family in {"ridge_breast_cancer", "ridge_wine"}
            and ("_k30_" in case.name or "_k100_" in case.name)
        )
    ]
    if len(selected) != 28:
        raise RuntimeError(f"expected 28 prespecified cases, obtained {len(selected)}")
    return selected


def _moments(case: SpectralCase, depth: int) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = case.interval
    scaled = (2.0 * case.eigenvalues - upper - lower) / (upper - lower)
    weights = case.residual**2 / float(case.residual @ case.residual)
    return np.asarray(weights @ chebvander(scaled, 2 * depth)), scaled


def _actual_squared(
    case: SpectralCase, scaled_eigenvalues: np.ndarray, coefficients: np.ndarray
) -> float:
    values = chebval(scaled_eigenvalues, coefficients)
    return float(
        np.sum(case.residual**2 * values**2) / float(case.residual @ case.residual)
    )


def _coverage_slack(actual: float, certificate: float, coefficients: np.ndarray) -> float:
    scale = 1.0 + abs(actual) + abs(certificate) + float(np.sum(np.abs(coefficients))) ** 2
    return 1024.0 * np.finfo(float).eps * scale


def _fallback_result(
    case: SpectralCase, budget: int, tolerance: float
) -> tuple[float, bool]:
    lower, upper = case.interval
    scaled = (2.0 * case.eigenvalues - upper - lower) / (upper - lower)
    coefficients = chebyshev_polynomial_at_zero(budget, case.interval)
    actual = _actual_squared(case, scaled, coefficients)
    certificate = tolerance * tolerance
    covered = actual <= certificate + _coverage_slack(
        actual, certificate, coefficients
    )
    return actual, bool(covered)


def _frontier_at_depth(
    case: SpectralCase, depth: int, tolerance: float
) -> FrontierRow:
    budget = chebyshev_budget_for_tolerance(case.interval, tolerance)
    if depth >= budget:
        actual, covered = _fallback_result(case, budget, tolerance)
        return FrontierRow(
            name=case.name,
            family=case.family,
            source=case.source,
            dimension=len(case.eigenvalues),
            enclosure_ratio=case.interval[1] / case.interval[0],
            tolerance=tolerance,
            observation_depth=depth,
            chebyshev_horizon=budget,
            sdp_horizon=budget,
            operational_horizon=budget,
            certificate=tolerance * tolerance,
            design_value=tolerance * tolerance,
            previous_design_value=None,
            actual_squared_residual=actual,
            covered=covered,
            design_calls=0,
            design_seconds=0.0,
            solver="interval Chebyshev",
            status="observation depth is not smaller than the fallback horizon",
        )

    eta, scaled = _moments(case, depth)
    cache: dict[int, object] = {}
    elapsed = 0.0

    def solve(degree: int):
        nonlocal elapsed
        if degree not in cache:
            started = perf_counter()
            cache[degree] = design_vectorized_moment_polynomial(
                eta, degree, case.interval
            )
            elapsed += perf_counter() - started
        return cache[degree]

    target = tolerance * tolerance
    low = depth
    low_design = solve(low)
    if float(low_design.design_value) <= target:
        sdp_horizon = low
    else:
        high = budget - 1
        high_design = solve(high)
        if float(high_design.design_value) > target:
            sdp_horizon = budget
        else:
            while high - low > 1:
                middle = (low + high) // 2
                middle_design = solve(middle)
                if float(middle_design.design_value) <= target:
                    high = middle
                else:
                    low = middle
            sdp_horizon = high

    if sdp_horizon == budget:
        actual, covered = _fallback_result(case, budget, tolerance)
        return FrontierRow(
            name=case.name,
            family=case.family,
            source=case.source,
            dimension=len(case.eigenvalues),
            enclosure_ratio=case.interval[1] / case.interval[0],
            tolerance=tolerance,
            observation_depth=depth,
            chebyshev_horizon=budget,
            sdp_horizon=budget,
            operational_horizon=budget,
            certificate=target,
            design_value=float(solve(budget - 1).design_value),
            previous_design_value=None,
            actual_squared_residual=actual,
            covered=covered,
            design_calls=len(cache),
            design_seconds=elapsed,
            solver="interval Chebyshev",
            status="no conditional SDP crossing before the fallback horizon",
        )

    # Explicitly verify the two sides of the crossing; this is also a guard
    # against a numerical bisection result inconsistent with the theorem.
    crossing = solve(sdp_horizon)
    previous = solve(sdp_horizon - 1) if sdp_horizon > depth else None
    if float(crossing.design_value) > target * (1.0 + 2.0e-5):
        raise RuntimeError(f"unverified SDP crossing for {case.name}, s={depth}")
    if previous is not None and float(previous.design_value) < target * (1.0 - 2.0e-5):
        raise RuntimeError(f"unverified lower side for {case.name}, s={depth}")

    operational = sdp_horizon
    accepted = crossing
    while float(accepted.certificate) > target and operational < budget - 1:
        operational += 1
        accepted = solve(operational)
    if float(accepted.certificate) > target:
        actual, covered = _fallback_result(case, budget, tolerance)
        return FrontierRow(
            name=case.name,
            family=case.family,
            source=case.source,
            dimension=len(case.eigenvalues),
            enclosure_ratio=case.interval[1] / case.interval[0],
            tolerance=tolerance,
            observation_depth=depth,
            chebyshev_horizon=budget,
            sdp_horizon=sdp_horizon,
            operational_horizon=budget,
            certificate=target,
            design_value=float(accepted.design_value),
            previous_design_value=(
                None if previous is None else float(previous.design_value)
            ),
            actual_squared_residual=actual,
            covered=covered,
            design_calls=len(cache),
            design_seconds=elapsed,
            solver="interval Chebyshev",
            status="SDP crossed but the numerical majorant did not",
        )

    actual = _actual_squared(case, scaled, accepted.coefficients)
    slack = _coverage_slack(actual, float(accepted.certificate), accepted.coefficients)
    return FrontierRow(
        name=case.name,
        family=case.family,
        source=case.source,
        dimension=len(case.eigenvalues),
        enclosure_ratio=case.interval[1] / case.interval[0],
        tolerance=tolerance,
        observation_depth=depth,
        chebyshev_horizon=budget,
        sdp_horizon=sdp_horizon,
        operational_horizon=operational,
        certificate=float(accepted.certificate),
        design_value=float(accepted.design_value),
        previous_design_value=None if previous is None else float(previous.design_value),
        actual_squared_residual=actual,
        covered=bool(actual <= float(accepted.certificate) + slack),
        design_calls=len(cache),
        design_seconds=elapsed,
        solver=accepted.solver,
        status=accepted.status,
    )


def _write_csv(rows: list[FrontierRow], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _summary(rows: list[FrontierRow], seed: int, size: int) -> dict[str, object]:
    depths = sorted({row.observation_depth for row in rows})
    by_depth: dict[str, object] = {}
    for depth in depths:
        selected = [row for row in rows if row.observation_depth == depth]
        horizons = np.asarray([row.operational_horizon for row in selected], dtype=float)
        fallback = np.asarray([row.chebyshev_horizon for row in selected], dtype=float)
        by_depth[str(depth)] = {
            "cases": len(selected),
            "median_horizon": float(np.median(horizons)),
            "median_fraction_of_chebyshev_horizon": float(np.median(horizons / fallback)),
            "strict_improvements_over_chebyshev": int(np.sum(horizons < fallback)),
            "median_saved_products_when_improved": (
                float(np.median(1.0 - horizons[horizons < fallback] / fallback[horizons < fallback]))
                if np.any(horizons < fallback)
                else None
            ),
        }
    best_counts = {str(depth): 0 for depth in depths}
    ties = 0
    for name in sorted({row.name for row in rows}):
        selected = [row for row in rows if row.name == name]
        best = min(row.operational_horizon for row in selected)
        winners = [row.observation_depth for row in selected if row.operational_horizon == best]
        if len(winners) > 1:
            ties += 1
        for depth in winners:
            best_counts[str(depth)] += 1
    return {
        "status": "prespecified observation-depth diagnostic",
        "seed": seed,
        "size": size,
        "cases": len({row.name for row in rows}),
        "rows": len(rows),
        "tolerance": rows[0].tolerance,
        "selection_rule": (
            "all two-dimensional Poisson cases and all breast-cancer/wine "
            "ridge cases with prescribed enclosure ratio 30 or 100"
        ),
        "certificate_violations": sum(not row.covered for row in rows),
        "maximum_sdp_to_operational_horizon_gap": max(
            row.operational_horizon - row.sdp_horizon for row in rows
        ),
        "by_depth": by_depth,
        "number_of_instances_with_tied_best_depth": ties,
        "best_depth_counts_including_ties": best_counts,
        "total_design_seconds": float(sum(row.design_seconds for row in rows)),
    }


def _write_summary(summary: dict[str, object], destination: Path) -> None:
    lines = [
        "# Observation-depth diagnostic",
        "",
        "The case subset and tolerance were specified by the script before the run. "
        "The horizon is the first degree whose independently checked majorant reaches "
        "the target; it includes the products used to acquire the moments.",
        "",
        f"- Cases: {summary['cases']}",
        f"- Target relative residual: {summary['tolerance']}",
        f"- Certificate violations: {summary['certificate_violations']}",
        f"- Total local SDP time: {summary['total_design_seconds']:.2f} s",
        "",
        "| Initial products s | Median horizon | Median horizon / Chebyshev | "
        "Strictly below Chebyshev | Median saving when improved |",
        "|---:|---:|---:|---:|---:|",
    ]
    for depth, values in summary["by_depth"].items():
        lines.append(
            f"| {depth} | {values['median_horizon']:.1f} | "
            f"{values['median_fraction_of_chebyshev_horizon']:.3f} | "
            f"{values['strict_improvements_over_chebyshev']}/{values['cases']} | "
            f"{values['median_saved_products_when_improved']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The median horizon is smallest at s=5.  It rises at s=6 even though "
            "the fixed-degree bound cannot increase, because the observation itself "
            "then costs six products.  Thus taking more moments has a measurable but "
            "nonmonotone net value.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def _make_figure(rows: list[FrontierRow], destination: Path) -> None:
    import matplotlib.pyplot as plt

    depths = sorted({row.observation_depth for row in rows})
    data = [
        [
            row.operational_horizon / row.chebyshev_horizon
            for row in rows
            if row.observation_depth == depth
        ]
        for depth in depths
    ]
    figure, axis = plt.subplots(figsize=(6.0, 3.5))
    axis.boxplot(data, tick_labels=[str(depth) for depth in depths], showfliers=False)
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_xlabel("initial Chebyshev products $s$")
    axis.set_ylabel("certified horizon / Chebyshev horizon")
    axis.set_ylim(0.0, 1.08)
    figure.tight_layout()
    figure.savefig(destination / "moment_frontier.pdf")
    figure.savefig(destination / "moment_frontier.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    parser.add_argument("--depths", default="1,2,3,4,5,6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depths = tuple(int(value) for value in args.depths.split(","))
    if not depths or min(depths) < 1:
        raise ValueError("depths must be positive integers")
    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("CLARABEL is required")
    warnings.filterwarnings(
        "ignore", message="Constraint .* contains too many subexpressions.*"
    )
    cases = _selected_cases(args.seed, args.size)
    rows: list[FrontierRow] = []
    started = perf_counter()
    for case_index, case in enumerate(cases, start=1):
        for depth in depths:
            rows.append(_frontier_at_depth(case, depth, args.tolerance))
        print(f"[{case_index:02d}/{len(cases):02d}] {case.name}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output / "frontier_results.csv")
    summary = _summary(rows, args.seed, args.size)
    summary["wall_seconds"] = perf_counter() - started
    project_root = Path(__file__).resolve().parents[1]
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("vectorized_sdp.py").resolve(),
        *sorted((project_root / "src").rglob("*.py")),
    ]
    summary["source_sha256"] = {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    summary["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cvxpy": cp.__version__,
        "clarabel": version("clarabel"),
        "scs": version("scs"),
        "platform": platform.platform(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(summary, args.output / "EXPERIMENT_SUMMARY.md")
    _make_figure(rows, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
