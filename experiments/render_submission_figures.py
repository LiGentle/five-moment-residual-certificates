"""Render the two manuscript figures from a frozen validation snapshot.

This script performs plotting only.  It reads the row-level CSV files produced
by ``run_validation.py`` and does not recompute any numerical result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = (
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

FAMILY_LABELS = {
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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def render(snapshot: Path, destination: Path) -> None:
    design_rows = _rows(snapshot / "design_results.csv")
    target_rows = _rows(snapshot / "target_results.csv")
    destination.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    degrees = sorted({int(row["degree"]) for row in design_rows})
    degree_data = [
        [
            min(1.2, float(row["certificate_to_chebyshev_bound"]))
            for row in design_rows
            if int(row["degree"]) == degree
        ]
        for degree in degrees
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.boxplot(
        degree_data,
        tick_labels=[str(value) for value in degrees],
        showfliers=False,
    )
    axis.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
    axis.set_xlabel("polynomial degree")
    axis.set_ylabel("bound ratio")
    axis.set_ylim(0.0, 1.2)
    figure.tight_layout()
    figure.savefig(
        destination / "certificate_improvement.pdf",
        bbox_inches="tight",
        pad_inches=0.18,
    )
    figure.savefig(
        destination / "certificate_improvement.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)

    tolerances = sorted(
        {float(row["tolerance"]) for row in target_rows}, reverse=True
    )
    families = [
        family
        for family in FAMILY_ORDER
        if any(row["family"] == family for row in target_rows)
    ]
    matrix = np.zeros((len(families), len(tolerances)))
    for row_index, family in enumerate(families):
        for column_index, tolerance in enumerate(tolerances):
            selected = [
                row
                for row in target_rows
                if row["family"] == family
                and float(row["tolerance"]) == tolerance
            ]
            matrix[row_index, column_index] = np.mean(
                [row["certificate_passed"].lower() == "true" for row in selected]
            )

    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(
        range(len(tolerances)), [f"{value:.0e}" for value in tolerances]
    )
    axis.set_yticks(
        range(len(families)), [FAMILY_LABELS[name] for name in families]
    )
    axis.set_xlabel("relative residual tolerance")
    for row_index in range(len(families)):
        for column_index in range(len(tolerances)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axis, label="fraction accepted early")
    figure.tight_layout()
    figure.savefig(
        destination / "target_activation.pdf", bbox_inches="tight", pad_inches=0.18
    )
    figure.savefig(
        destination / "target_activation.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    render(arguments.snapshot, arguments.output)
