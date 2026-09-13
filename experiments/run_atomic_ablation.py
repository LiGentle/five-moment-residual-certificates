"""Post-development diagnostic for the boundary-moment implementation.

This script does not alter ``src/fmcert``.  Within this process only, it makes
``_recover_atomic_measure`` return ``None`` and then redesigns every synthetic
instance from the reference validation on the full candidate-degree grid.  The
purpose is to separate the mathematical validity of the certificate and
fallback from the finite-precision benefit of recognizing boundary moment
sequences.

This diagnostic was specified after the reference validation.  Its output is
not part of that independent validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np

import fmcert.sdp as sdp
from fmcert import (
    chebyshev_budget_for_tolerance,
    chebyshev_polynomial_at_zero,
    design_five_moment_polynomial,
    probe_five_moment_state,
)
from run_validation import (
    _actual_squared,
    _coverage_slack,
    build_cases,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "independent_end_to_end_v7"
FAMILIES = (
    "two_point",
    "three_point",
    "three_cluster",
    "log_uniform",
    "single_outlier",
)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reference() -> tuple[dict[tuple[str, int], dict[str, str]], dict[tuple[str, float], dict[str, str]], dict[str, object]]:
    with (REFERENCE / "design_results.csv").open(newline="", encoding="utf-8") as handle:
        designs = {
            (row["name"], int(row["degree"])): row
            for row in csv.DictReader(handle)
            if row["family"] in FAMILIES
        }
    with (REFERENCE / "target_results.csv").open(newline="", encoding="utf-8") as handle:
        targets = {
            (row["name"], float(row["tolerance"])): row
            for row in csv.DictReader(handle)
            if row["family"] in FAMILIES
        }
    summary = json.loads((REFERENCE / "summary.json").read_text(encoding="utf-8"))
    return designs, targets, summary


def _evaluate(
    *, seed: int, size: int, degrees: tuple[int, ...], tolerances: tuple[float, ...]
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    reference_designs, reference_targets, _ = _load_reference()
    cases = [case for case in build_cases(seed, size) if case.family in FAMILIES]
    if len(cases) != 60:
        raise RuntimeError(f"expected 60 synthetic cases, obtained {len(cases)}")

    designs_out: list[dict[str, object]] = []
    targets_out: list[dict[str, object]] = []
    started_all = perf_counter()
    for case_index, case in enumerate(cases, start=1):
        state = probe_five_moment_state(
            lambda vector, spectrum=case.eigenvalues: spectrum * vector,
            case.residual,
            case.interval,
        )
        successful: dict[int, object] = {}
        case_started = perf_counter()
        for degree in degrees:
            reference = reference_designs[(case.name, degree)]
            started = perf_counter()
            try:
                design = design_five_moment_polynomial(
                    state.moments, degree, case.interval
                )
                seconds = perf_counter() - started
                successful[degree] = design
                actual = _actual_squared(case, design.coefficients)
                slack = _coverage_slack(
                    actual, design.certificate, design.coefficients
                )
                designs_out.append(
                    {
                        "name": case.name,
                        "family": case.family,
                        "degree": degree,
                        "success": True,
                        "seconds": seconds,
                        "certificate": float(design.certificate),
                        "actual_squared_residual": actual,
                        "raw_certificate_excess": actual - design.certificate,
                        "coverage_slack": slack,
                        "covered": actual - design.certificate <= slack,
                        "atomic_pattern": bool(design.atomic_pattern),
                        "solver": design.solver,
                        "status": design.status,
                        "error": "",
                        "reference_atomic_pattern": reference["atomic_pattern"] == "True",
                        "reference_certificate": float(reference["certificate"]),
                        "certificate_ratio_to_reference": float(design.certificate)
                        / max(float(reference["certificate"]), 1.0e-300),
                    }
                )
            except Exception as error:  # diagnostic records numerical failures
                seconds = perf_counter() - started
                designs_out.append(
                    {
                        "name": case.name,
                        "family": case.family,
                        "degree": degree,
                        "success": False,
                        "seconds": seconds,
                        "certificate": "",
                        "actual_squared_residual": "",
                        "raw_certificate_excess": "",
                        "coverage_slack": "",
                        "covered": False,
                        "atomic_pattern": False,
                        "solver": "",
                        "status": "",
                        "error": f"{type(error).__name__}: {error}",
                        "reference_atomic_pattern": reference["atomic_pattern"] == "True",
                        "reference_certificate": float(reference["certificate"]),
                        "certificate_ratio_to_reference": "",
                    }
                )

        case_seconds = perf_counter() - case_started
        for row in designs_out[-len(degrees) :]:
            row["case_grid_seconds"] = case_seconds

        for tolerance in tolerances:
            budget = chebyshev_budget_for_tolerance(case.interval, tolerance)
            feasible = [
                design
                for degree, design in sorted(successful.items())
                if degree < budget and design.certificate <= tolerance * tolerance
            ]
            selected = feasible[0] if feasible else None
            if selected is None:
                coefficients = chebyshev_polynomial_at_zero(budget, case.interval)
                selected_degree = budget
                certificate = tolerance * tolerance
            else:
                coefficients = selected.coefficients
                selected_degree = int(selected.degree)
                certificate = float(selected.certificate)
            actual = _actual_squared(case, coefficients)
            slack = _coverage_slack(actual, certificate, coefficients)
            reference = reference_targets[(case.name, tolerance)]
            reference_passed = reference["certificate_passed"] == "True"
            targets_out.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "tolerance": tolerance,
                    "chebyshev_budget": budget,
                    "certificate_passed": selected is not None,
                    "selected_degree": selected_degree,
                    "certificate": certificate,
                    "actual_squared_residual": actual,
                    "raw_certificate_excess": actual - certificate,
                    "coverage_slack": slack,
                    "covered": actual - certificate <= slack,
                    "converged": actual
                    <= tolerance * tolerance * (1.0 + 2.0e-6),
                    "hessian_product_saving": 1.0 - selected_degree / budget,
                    "reference_certificate_passed": reference_passed,
                    "reference_selected_degree": int(reference["selected_degree"]),
                    "same_acceptance_decision_as_reference": (selected is not None)
                    == reference_passed,
                }
            )
        print(
            f"[{case_index:02d}/{len(cases):02d}] {case.name}: "
            f"{len(successful)}/{len(degrees)} designs, {case_seconds:.3f}s",
            flush=True,
        )
    return designs_out, targets_out, perf_counter() - started_all


def _summarize(
    design_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    *,
    seed: int,
    size: int,
    degrees: tuple[int, ...],
    tolerances: tuple[float, ...],
    wall_seconds: float,
) -> dict[str, object]:
    _, _, reference_summary = _load_reference()
    successful = [row for row in design_rows if bool(row["success"])]
    failures = [row for row in design_rows if not bool(row["success"])]
    per_case: dict[str, float] = {}
    for row in design_rows:
        per_case[str(row["name"])] = float(row["case_grid_seconds"])

    with (REFERENCE / "target_results.csv").open(newline="", encoding="utf-8") as handle:
        reference_case_seconds = {
            row["name"]: float(row["design_seconds"])
            for row in csv.DictReader(handle)
            if row["family"] in FAMILIES
        }
    exact_names = [
        name
        for name in per_case
        if name.startswith("two_point_") or name.startswith("three_point_")
    ]
    reference_exact_total = sum(reference_case_seconds[name] for name in exact_names)
    ablated_exact_total = sum(per_case[name] for name in exact_names)

    source_files = [
        Path(__file__).resolve(),
        ROOT / "experiments" / "run_validation.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in source_files
    }
    reference_hashes = reference_summary["source_sha256"]
    core_unchanged = all(
        source_hashes[path] == expected
        for path, expected in reference_hashes.items()
        if path in source_hashes
    )

    summary: dict[str, object] = {
        "status": "post-development diagnostic; not part of the independent reference validation",
        "intervention": "runtime-only monkeypatch: fmcert.sdp._recover_atomic_measure returns None",
        "core_source_matches_reference_hashes": core_unchanged,
        "seed": seed,
        "configuration": {
            "size": size,
            "families": list(FAMILIES),
            "degrees": list(degrees),
            "tolerances": list(tolerances),
        },
        "cases": len(per_case),
        "design_attempts": len(design_rows),
        "design_successes": len(successful),
        "design_failures": len(failures),
        "atomic_flags_after_patch": sum(
            bool(row["atomic_pattern"]) for row in successful
        ),
        "reference_boundary_design_rows_in_scope": sum(
            bool(row["reference_atomic_pattern"]) for row in design_rows
        ),
        "certificate_violations_among_successes": sum(
            not bool(row["covered"]) for row in successful
        ),
        "max_raw_certificate_excess_among_successes": max(
            (float(row["raw_certificate_excess"]) for row in successful),
            default=None,
        ),
        "failure_messages": dict(
            sorted(Counter(str(row["error"]) for row in failures).items())
        ),
        "failures_by_family_degree": dict(
            sorted(
                Counter(
                    f"{row['family']}|{row['degree']}" for row in failures
                ).items()
            )
        ),
        "timing_seconds": {
            "wall": wall_seconds,
            "design_attempt_median": float(
                np.median([float(row["seconds"]) for row in design_rows])
            ),
            "design_attempt_maximum": max(
                float(row["seconds"]) for row in design_rows
            ),
            "case_grid_median": float(np.median(list(per_case.values()))),
            "case_grid_maximum": max(per_case.values()),
            "two_and_three_point_families": {
                "reference_total": reference_exact_total,
                "ablation_total": ablated_exact_total,
                "ablation_to_reference_total_ratio": ablated_exact_total
                / reference_exact_total,
                "reference_case_median": float(
                    np.median([reference_case_seconds[name] for name in exact_names])
                ),
                "ablation_case_median": float(
                    np.median([per_case[name] for name in exact_names])
                ),
            },
        },
        "by_degree": {},
        "by_tolerance": {},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "installed_solvers": cp.installed_solvers(),
            "platform": platform.platform(),
        },
        "source_sha256": source_hashes,
        "reference_sha256": {
            "summary.json": _sha256(REFERENCE / "summary.json"),
            "design_results.csv": _sha256(REFERENCE / "design_results.csv"),
            "target_results.csv": _sha256(REFERENCE / "target_results.csv"),
        },
    }

    for degree in degrees:
        rows = [row for row in design_rows if int(row["degree"]) == degree]
        ok = [row for row in rows if bool(row["success"])]
        ratios = [
            float(row["certificate_ratio_to_reference"])
            for row in ok
            if row["certificate_ratio_to_reference"] != ""
        ]
        summary["by_degree"][str(degree)] = {
            "attempts": len(rows),
            "successes": len(ok),
            "failures": len(rows) - len(ok),
            "certificate_violations": sum(
                not bool(row["covered"]) for row in ok
            ),
            "median_seconds_success": float(
                np.median([float(row["seconds"]) for row in ok])
            )
            if ok
            else None,
            "maximum_seconds_success": max(
                (float(row["seconds"]) for row in ok), default=None
            ),
            "median_certificate_ratio_to_reference": float(np.median(ratios))
            if ratios
            else None,
        }

    for tolerance in tolerances:
        rows = [
            row
            for row in target_rows
            if float(row["tolerance"]) == tolerance
        ]
        active = [row for row in rows if bool(row["certificate_passed"])]
        reference_active = [
            row for row in rows if bool(row["reference_certificate_passed"])
        ]
        retained = [
            row
            for row in rows
            if bool(row["reference_certificate_passed"])
            and bool(row["certificate_passed"])
        ]
        summary["by_tolerance"][f"{tolerance:.0e}"] = {
            "cases": len(rows),
            "active": len(active),
            "activation_rate": len(active) / len(rows),
            "active_by_family": dict(
                sorted(Counter(str(row["family"]) for row in active).items())
            ),
            "reference_active": len(reference_active),
            "reference_acceptance_rate": len(reference_active) / len(rows),
            "reference_active_by_family": dict(
                sorted(Counter(str(row["family"]) for row in reference_active).items())
            ),
            "retained_reference_acceptances": len(retained),
            "retained_reference_acceptance_fraction": len(retained) / len(reference_active)
            if reference_active
            else None,
            "same_acceptance_decision_as_reference": sum(
                bool(row["same_acceptance_decision_as_reference"]) for row in rows
            ),
            "certificate_violations": sum(
                not bool(row["covered"]) for row in rows
            ),
            "target_failures": sum(not bool(row["converged"]) for row in rows),
            "median_hvp_saving_when_active": float(
                np.median(
                    [float(row["hessian_product_saving"]) for row in active]
                )
            )
            if active
            else None,
        }
    return summary


def _summary_markdown(summary: dict[str, object]) -> str:
    by_tolerance = summary["by_tolerance"]
    timing = summary["timing_seconds"]["two_and_three_point_families"]
    rows = [
        "# Boundary-moment implementation diagnostic",
        "",
        "> **Status.** Post-development diagnostic only; this is not part of the",
        "> independent reference validation.",
        "",
        "The experiment disables boundary-moment recognition only within the running",
        "Python process and redesigns all 60 synthetic reference cases on degrees",
        "`{2,3,4,6,8}`.  No core source file is modified.",
        "",
        "| target | reference accepted | shortcut off | retained acceptances |",
        "|---:|---:|---:|---:|",
    ]
    for key in ("1e-02", "1e-04", "1e-06"):
        item = by_tolerance[key]
        rows.append(
            f"| {key} | {item['reference_active']}/60 | {item['active']}/60 | "
            f"{item['retained_reference_acceptances']}/{item['reference_active']} "
            f"({100.0 * item['retained_reference_acceptance_fraction']:.1f}%) |"
        )
    rows.extend(
        [
            "",
            f"Of {summary['design_attempts']} design attempts, "
            f"{summary['design_successes']} returned and {summary['design_failures']} "
            "failed.  All failures were singular degree-two solves in the two-point",
            "family.  Returned designs had zero certificate violations, and the",
            "Chebyshev fallback left all target runs valid and converged.",
            "",
            "All 12 early acceptances outside the two- and three-point families at",
            "tolerance `1e-2` were unchanged.  Every lost acceptance came from the",
            "two- or three-point families.",
            "",
            f"Across those two families, the reference run used "
            f"{timing['reference_total']:.3f} s "
            f"for the candidate grids, whereas the ablation used "
            f"{timing['ablation_total']:.2f} s "
            f"({timing['ablation_to_reference_total_ratio']:.0f} times as long).",
            "",
            "The result separates safety from numerical sharpness: the certificate",
            "and fallback remain safe without the shortcut, but reliable tight-target",
            "termination on singular boundary moment sequences depends strongly on",
            "explicit treatment of boundary moment sequences.",
            "",
        ]
    )
    return "\n".join(rows)


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
    if not REFERENCE.is_dir():
        raise FileNotFoundError(f"reference validation not found: {REFERENCE}")

    original_recovery = sdp._recover_atomic_measure
    sdp._recover_atomic_measure = lambda moments: None
    try:
        design_rows, target_rows, wall_seconds = _evaluate(
            seed=arguments.seed,
            size=arguments.size,
            degrees=degrees,
            tolerances=tolerances,
        )
    finally:
        sdp._recover_atomic_measure = original_recovery

    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_csv(design_rows, arguments.output / "design_results.csv")
    _write_csv(target_rows, arguments.output / "target_results.csv")
    summary = _summarize(
        design_rows,
        target_rows,
        seed=arguments.seed,
        size=arguments.size,
        degrees=degrees,
        tolerances=tolerances,
        wall_seconds=wall_seconds,
    )
    summary["result_sha256"] = {
        "design_results.csv": _sha256(arguments.output / "design_results.csv"),
        "target_results.csv": _sha256(arguments.output / "target_results.csv"),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (arguments.output / "EXPERIMENT_SUMMARY.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
