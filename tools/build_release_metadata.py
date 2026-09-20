"""Build traceable metadata and SHA-256 manifest for release version 2.0.0."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mpc_timing_snapshot() -> dict[str, object]:
    directory = RESULTS / "constrained_mpc_v1"
    summary = load_json(directory / "summary.json")
    with (directory / "timing.csv").open(newline="", encoding="utf-8") as stream:
        timing = list(csv.DictReader(stream))
    with (directory / "systems.csv").open(newline="", encoding="utf-8") as stream:
        systems = list(csv.DictReader(stream))

    aggregate: dict[str, object] = {}
    for method in ("certificate", "interval_chebyshev", "cg", "direct"):
        values = [
            int(row["total_ns"]) / 1.0e3
            for row in timing
            if row["method"] == method
        ]
        aggregate[method] = {
            "median_microseconds": statistics.median(values),
            "mean_microseconds": statistics.mean(values),
        }

    system_keys = {
        int(row["system"]): (
            row["plant"],
            int(row["horizon"]),
            int(row["state_index"]),
        )
        for row in systems
    }
    by_path: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for row in timing:
        key = system_keys[int(row["system"])]
        by_method = by_path.setdefault(key, {})
        by_method.setdefault(row["method"], []).append(int(row["total_ns"]) / 1.0e3)
    path_wins = sum(
        statistics.mean(values["certificate"])
        < statistics.mean(values["interval_chebyshev"])
        for values in by_path.values()
    )

    configurations = []
    for item in summary["configurations"]:
        cert = item["by_method"]["certificate"]
        cheb = item["by_method"]["interval_chebyshev"]
        configurations.append(
            {
                "plant": item["plant"],
                "horizon": item["horizon"],
                "systems": item["newton_systems"],
                "certificate_median_microseconds": cert["median_microseconds"],
                "chebyshev_median_microseconds": cheb["median_microseconds"],
                "certificate_mean_microseconds": cert["mean_microseconds"],
                "chebyshev_mean_microseconds": cheb["mean_microseconds"],
            }
        )
    return {
        "aggregate": aggregate,
        "configurations": configurations,
        "certificate_mean_path_wins": path_wins,
        "paths": len(by_path),
    }


def build_metadata() -> None:
    main_dir = RESULTS / "independent_end_to_end_v7"
    main_summary = load_json(main_dir / "summary.json")
    frontier_summary = load_json(RESULTS / "moment_frontier_v2" / "summary.json")
    boundary_summary = load_json(
        RESULTS / "boundary_moment_diagnostic_v2" / "summary.json"
    )
    mpc_e2e = load_json(RESULTS / "constrained_mpc_v1" / "e2e_summary.json")
    jacobi = load_json(RESULTS / "constrained_mpc_jacobi_pilot" / "summary.json")

    figure_specs = {
        "certificate_improvement": {
            "data": main_dir / "design_results.csv",
            "pdf": main_dir / "certificate_improvement.pdf",
            "png": main_dir / "certificate_improvement.png",
        },
        "target_activation": {
            "data": main_dir / "target_results.csv",
            "pdf": main_dir / "target_activation.pdf",
            "png": main_dir / "target_activation.png",
        },
    }
    figure_provenance: dict[str, object] = {
        "generator": "experiments/render_submission_figures.py",
        "generator_sha256": sha256(
            ROOT / "experiments" / "render_submission_figures.py"
        ),
        "frozen_run": "results/independent_end_to_end_v7",
        "figures": {},
    }
    for name, files in figure_specs.items():
        figure_provenance["figures"][name] = {
            "data_file": str(files["data"].relative_to(ROOT)),
            "data_sha256": sha256(files["data"]),
            "pdf_file": str(files["pdf"].relative_to(ROOT)),
            "pdf_sha256": sha256(files["pdf"]),
            "png_file": str(files["png"].relative_to(ROOT)),
            "png_sha256": sha256(files["png"]),
        }
    write_json(ROOT / "FIGURE_PROVENANCE.json", figure_provenance)

    frozen_runs = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_status": "version 2.0.0 release package",
        "environment": main_summary["environment"],
        "timing_policy": (
            "Frozen observations from one local run; machine-, load-, library-, "
            "and run-dependent; not a correctness criterion."
        ),
        "main": {
            "directory": "results/independent_end_to_end_v7",
            "candidate_grid_design_seconds": main_summary[
                "candidate_grid_design_seconds"
            ],
            "wall_seconds": main_summary["wall_seconds"],
        },
        "frontier": {
            "directory": "results/moment_frontier_v2",
            "total_design_seconds": frontier_summary["total_design_seconds"],
            "wall_seconds": frontier_summary["wall_seconds"],
        },
        "boundary": {
            "directory": "results/boundary_moment_diagnostic_v2",
            "timing_seconds": boundary_summary["timing_seconds"],
        },
        "mpc": {
            "directory": "results/constrained_mpc_v1",
            "timing": mpc_timing_snapshot(),
            "end_to_end_products": {
                method: mpc_e2e["by_method"][method]["total_matrix_products"]
                for method in ("certificate", "interval_chebyshev", "cg")
            },
        },
        "jacobi": {
            "directory": "results/constrained_mpc_jacobi_pilot",
            "certificate_product_ratio_to_interval_chebyshev": jacobi["end_to_end"][
                "certificate_product_ratio_to_interval_chebyshev"
            ],
        },
    }
    write_json(ROOT / "FROZEN_RUNS.json", frozen_runs)

    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".DS_Store"}
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not any(part in excluded_parts for part in path.parts)
    ]
    lines = [
        f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}"
        for path in sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())
    ]
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_metadata()
