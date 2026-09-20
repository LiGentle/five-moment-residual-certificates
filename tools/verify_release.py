"""Verify v2.0.0 results, provenance hashes, figures, and archive manifest."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKS: list[dict[str, object]] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def record(name: str, passed: bool, detail: object) -> None:
    CHECKS.append({"name": name, "passed": bool(passed), "detail": detail})


def verify_hash_map(label: str, values: dict[str, str]) -> None:
    for relative, expected in sorted(values.items()):
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        record(f"{label}:{relative}", actual == expected, {"expected": expected, "actual": actual})


def close(actual: float, expected: float, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def verify_science() -> None:
    main = load_json(RESULTS / "independent_end_to_end_v7" / "summary.json")
    expected_main = {
        "2": (0.0641, 1.00),
        "3": (0.0422, 1.49),
        "4": (0.0476, 2.18),
        "6": (0.0553, 1.88),
        "8": (0.0588, 1.85),
    }
    for degree, (cheb, actual) in expected_main.items():
        item = main["by_degree"][degree]
        record(
            f"main:degree-{degree}",
            round(item["median_certificate_to_chebyshev_bound"], 4) == cheb
            and round(item["median_certificate_to_actual_nonzero"], 2) == actual
            and close(item["fraction_strictly_below_chebyshev_bound"], 1.0),
            item,
        )
    record(
        "main:coverage",
        main["design_checks"] == 540
        and main["target_runs"] == 324
        and main["robust_checks"] == 36
        and main["certificate_violations"] == 0
        and main["target_certificate_violations"] == 0
        and main["robust_violations"] == 0,
        {
            "design_checks": main["design_checks"],
            "target_runs": main["target_runs"],
            "robust_checks": main["robust_checks"],
        },
    )
    expected_environment = {
        "python": "3.11.15",
        "numpy": "2.4.6",
        "cvxpy": "1.9.2",
        "clarabel": "0.11.1",
        "scs": "3.2.11",
        "scikit_learn": "1.9.0",
        "matplotlib": "3.11.0",
    }
    record(
        "main:pinned-environment",
        all(main["environment"].get(key) == value for key, value in expected_environment.items()),
        main["environment"],
    )

    frontier = load_json(RESULTS / "moment_frontier_v2" / "summary.json")
    expected_horizons = {"1": 11.0, "2": 9.5, "3": 8.5, "4": 7.0, "5": 5.0, "6": 6.0}
    record(
        "frontier:science",
        frontier["rows"] == 168
        and frontier["certificate_violations"] == 0
        and frontier["maximum_sdp_to_operational_horizon_gap"] == 1
        and all(close(frontier["by_depth"][key]["median_horizon"], value) for key, value in expected_horizons.items()),
        frontier["by_depth"],
    )

    boundary = load_json(RESULTS / "boundary_moment_diagnostic_v2" / "summary.json")
    record(
        "boundary:science",
        boundary["design_attempts"] == 300
        and boundary["design_successes"] == 296
        and boundary["design_failures"] == 4
        and boundary["certificate_violations_among_successes"] == 0
        and [boundary["by_tolerance"][key]["active"] for key in ("1e-02", "1e-04", "1e-06")] == [34, 8, 1]
        and boundary["core_source_matches_reference_hashes"] is True,
        boundary["by_tolerance"],
    )

    e2e = load_json(RESULTS / "constrained_mpc_v1" / "e2e_summary.json")
    cert = e2e["by_method"]["certificate"]
    cheb = e2e["by_method"]["interval_chebyshev"]
    record(
        "mpc:end-to-end",
        cert["converged_qps"] == 48
        and cheb["converged_qps"] == 48
        and cert["total_newton_updates"] == 1409
        and cheb["total_newton_updates"] == 1426
        and cert["total_matrix_products"] == 510169
        and cheb["total_matrix_products"] == 1200219
        and cert["maximum_linear_relative_residual"] <= 0.05,
        {"certificate": cert, "interval_chebyshev": cheb},
    )

    jacobi = load_json(RESULTS / "constrained_mpc_jacobi_pilot" / "summary.json")
    ratio = jacobi["end_to_end"]["certificate_product_ratio_to_interval_chebyshev"]
    record("jacobi:product-reduction", round(1.0 - ratio, 3) == 0.386, {"ratio": ratio})


def verify_provenance() -> None:
    main = load_json(RESULTS / "independent_end_to_end_v7" / "summary.json")
    frontier = load_json(RESULTS / "moment_frontier_v2" / "summary.json")
    boundary = load_json(RESULTS / "boundary_moment_diagnostic_v2" / "summary.json")
    mpc = load_json(RESULTS / "constrained_mpc_v1" / "summary.json")
    jacobi_protocol = load_json(
        RESULTS / "constrained_mpc_jacobi_pilot" / "protocol.json"
    )
    jacobi_summary = load_json(
        RESULTS / "constrained_mpc_jacobi_pilot" / "summary.json"
    )

    verify_hash_map("main-source", main["source_sha256"])
    verify_hash_map("frontier-source", frontier["source_sha256"])
    verify_hash_map("boundary-source", boundary["source_sha256"])
    verify_hash_map("mpc-source", mpc["source_sha256"])
    verify_hash_map("jacobi-source", jacobi_protocol["source_sha256"])
    record(
        "jacobi:protocol-summary-source-map",
        jacobi_protocol["source_sha256"]
        == jacobi_summary["protocol"]["source_sha256"],
        jacobi_protocol["source_sha256"],
    )

    main_dir = RESULTS / "independent_end_to_end_v7"
    for relative, expected in boundary["reference_sha256"].items():
        actual = sha256(main_dir / relative)
        record(
            f"boundary-reference:{relative}",
            actual == expected,
            {"expected": expected, "actual": actual},
        )
    boundary_dir = RESULTS / "boundary_moment_diagnostic_v2"
    for relative, expected in boundary["result_sha256"].items():
        actual = sha256(boundary_dir / relative)
        record(
            f"boundary-result:{relative}",
            actual == expected,
            {"expected": expected, "actual": actual},
        )

    figure_metadata = load_json(ROOT / "FIGURE_PROVENANCE.json")
    generator_actual = sha256(
        ROOT / "experiments" / "render_submission_figures.py"
    )
    record(
        "figures:generator",
        generator_actual == figure_metadata["generator_sha256"],
        {"expected": figure_metadata["generator_sha256"], "actual": generator_actual},
    )
    for name, values in figure_metadata["figures"].items():
        for kind in ("data", "pdf", "png"):
            actual = sha256(ROOT / values[f"{kind}_file"])
            expected = values[f"{kind}_sha256"]
            record(
                f"figure:{name}:{kind}",
                actual == expected,
                {"expected": expected, "actual": actual},
            )


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS"
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        relative = relative.removeprefix("./")
        listed.add(relative)
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        record(
            f"manifest:{relative}",
            actual == expected,
            {"expected": expected, "actual": actual},
        )
    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".DS_Store"}
    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not any(part in excluded_parts for part in path.parts)
    }
    record(
        "manifest:complete-file-set",
        listed == actual_files,
        {"missing": sorted(actual_files - listed), "extra": sorted(listed - actual_files)},
    )


def main() -> None:
    verify_science()
    verify_provenance()
    verify_manifest()
    failures = [item for item in CHECKS if not item["passed"]]
    report = {
        "status": "pass" if not failures else "fail",
        "checks": len(CHECKS),
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
