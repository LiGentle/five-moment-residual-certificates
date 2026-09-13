"""Five-moment certified polynomial iteration for SPD quadratics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from numpy.polynomial.chebyshev import chebdiv

from .moments import FiveMomentState, recover_five_moment_state
from .sdp import (
    MomentPolynomial,
    chebyshev_polynomial_at_zero,
    design_five_moment_polynomial,
)

Array = np.ndarray


@dataclass(frozen=True)
class CertifiedSolve:
    """Output of one certified polynomial solve."""

    residual: Array
    point: Array | None
    state: FiveMomentState | None
    design: MomentPolynomial | None
    polynomial_coefficients: Array
    hessian_products: int
    chebyshev_budget: int
    collective_reductions: int
    certificate_passed: bool
    converged: bool
    relative_residual: float
    certified_squared_relative_residual: float
    design_seconds: float


def chebyshev_budget_for_tolerance(
    spectral_interval: tuple[float, float], tolerance: float
) -> int:
    """Smallest degree whose interval Chebyshev bound reaches ``tolerance``."""

    lower, upper = (float(value) for value in spectral_interval)
    if not (0.0 < lower < upper and 0.0 < tolerance < 1.0):
        raise ValueError("invalid spectral interval or tolerance")
    outside = (upper + lower) / (upper - lower)
    return int(np.ceil(np.arccosh(1.0 / tolerance) / np.arccosh(outside)))


def _scaled_operator(
    hessian_product: Callable[[Array], Array],
    spectral_interval: tuple[float, float],
) -> Callable[[Array], Array]:
    lower, upper = spectral_interval
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)

    def product(vector: Array) -> Array:
        return (np.asarray(hessian_product(vector), dtype=float) - center * vector) / radius

    return product


def probe_five_moment_state(
    hessian_product: Callable[[Array], Array],
    residual: Array,
    spectral_interval: tuple[float, float],
) -> FiveMomentState:
    """Run the two-product shifted-Chebyshev probe and recover its moments."""

    g0 = np.asarray(residual, dtype=float)
    if g0.ndim != 1 or np.any(~np.isfinite(g0)) or float(g0 @ g0) <= 0.0:
        raise ValueError("residual must be a finite nonzero vector")
    scaled_product = _scaled_operator(hessian_product, spectral_interval)
    t1g = np.asarray(scaled_product(g0), dtype=float)
    t2g = 2.0 * np.asarray(scaled_product(t1g), dtype=float) - g0
    return recover_five_moment_state(
        g0, t1g, t2g, spectral_interval=spectral_interval
    )


def evaluate_with_cached_products(
    hessian_product: Callable[[Array], Array],
    residual: Array,
    state: FiveMomentState,
    coefficients: Array,
) -> tuple[Array, list[Array]]:
    """Evaluate a Chebyshev polynomial, reusing the two products in ``state``."""

    values = np.asarray(coefficients, dtype=float)
    degree = len(values) - 1
    if degree < 2:
        raise ValueError("the certified method uses degree at least two")
    basis = [
        np.asarray(residual, dtype=float),
        state.chebyshev_gradient,
        state.second_chebyshev_gradient,
    ]
    scaled_product = _scaled_operator(hessian_product, state.spectral_interval)
    while len(basis) <= degree:
        basis.append(2.0 * scaled_product(basis[-1]) - basis[-2])
    output = sum(float(values[index]) * basis[index] for index in range(degree + 1))
    return np.asarray(output, dtype=float), basis


def correction_chebyshev_coefficients(
    residual_coefficients: Array,
    spectral_interval: tuple[float, float],
) -> Array:
    """Return ``q`` such that ``p(H)g=g-H q(H)g`` in scaled coordinates."""

    lower, upper = spectral_interval
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)
    numerator = -np.asarray(residual_coefficients, dtype=float).copy()
    numerator[0] += 1.0
    quotient, remainder = chebdiv(numerator, np.asarray([center, radius]))
    scale = max(1.0, float(np.sum(np.abs(numerator))))
    if float(np.max(np.abs(remainder))) > 1.0e-10 * scale:
        raise ValueError("residual polynomial is not normalized at lambda=0")
    return np.asarray(np.trim_zeros(quotient, trim="b"), dtype=float)


def run_five_moment_method(
    *,
    hessian_product: Callable[[Array], Array],
    residual: Array,
    spectral_interval: tuple[float, float],
    tolerance: float = 1.0e-4,
    candidate_degrees: Iterable[int] | None = None,
    point: Array | None = None,
    moment_radii: Array | None = None,
    preferred_solver: str = "CLARABEL",
) -> CertifiedSolve:
    """Run the target-driven five-moment method.

    Two matrix products form ``T_0(Z)g,T_1(Z)g,T_2(Z)g`` directly, where ``Z``
    is the shifted and scaled Hessian.  The method then searches degrees in
    increasing order and accepts the first polynomial whose five-moment
    certificate reaches the requested residual tolerance.  If no candidate
    passes, it evaluates the classical degree-``N`` Chebyshev residual.  The
    cached products are reused, so the total number of matrix products equals
    the selected polynomial degree and never exceeds ``N``.
    """

    g0 = np.asarray(residual, dtype=float)
    if g0.ndim != 1 or np.any(~np.isfinite(g0)) or float(g0 @ g0) <= 0.0:
        raise ValueError("residual must be a finite nonzero vector")
    if point is not None and np.asarray(point).shape != g0.shape:
        raise ValueError("point and residual must be aligned")
    initial_norm = float(np.linalg.norm(g0))
    budget = chebyshev_budget_for_tolerance(spectral_interval, tolerance)
    if budget < 2:
        coefficients = chebyshev_polynomial_at_zero(budget, spectral_interval)
        scaled_product = _scaled_operator(hessian_product, spectral_interval)
        basis = [g0]
        while len(basis) <= budget:
            if len(basis) == 1:
                basis.append(np.asarray(scaled_product(basis[-1]), dtype=float))
            else:  # Defensive: the standard relative target gives budget 0 or 1 here.
                basis.append(2.0 * scaled_product(basis[-1]) - basis[-2])
        final_residual = sum(
            float(coefficients[index]) * basis[index]
            for index in range(len(coefficients))
        )
        final_point = None
        if point is not None:
            correction = correction_chebyshev_coefficients(
                coefficients, spectral_interval
            )
            correction_vector = sum(
                float(correction[index]) * basis[index]
                for index in range(len(correction))
            )
            final_point = np.asarray(point, dtype=float) - correction_vector
        ratio = float(np.linalg.norm(final_residual)) / initial_norm
        return CertifiedSolve(
            residual=np.asarray(final_residual, dtype=float),
            point=final_point,
            state=None,
            design=None,
            polynomial_coefficients=coefficients,
            hessian_products=budget,
            chebyshev_budget=budget,
            collective_reductions=0,
            certificate_passed=False,
            converged=bool(ratio <= tolerance * (1.0 + 2.0e-6)),
            relative_residual=ratio,
            certified_squared_relative_residual=tolerance * tolerance,
            design_seconds=0.0,
        )
    state = probe_five_moment_state(
        hessian_product,
        g0,
        spectral_interval,
    )
    if candidate_degrees is None:
        degrees = range(2, budget)
    else:
        degrees = sorted(
            {int(value) for value in candidate_degrees if 2 <= int(value) < budget}
        )

    chosen: MomentPolynomial | None = None
    elapsed = 0.0
    for degree in degrees:
        start = perf_counter()
        try:
            design = design_five_moment_polynomial(
                state.moments,
                degree,
                spectral_interval,
                moment_radii=moment_radii,
                preferred_solver=preferred_solver,
            )
        except RuntimeError:
            elapsed += perf_counter() - start
            continue
        elapsed += perf_counter() - start
        if design.certificate <= tolerance * tolerance:
            chosen = design
            break

    passed = chosen is not None
    if passed:
        assert chosen is not None
        coefficients = chosen.coefficients
        certified = chosen.certificate
        degree = chosen.degree
    else:
        coefficients = chebyshev_polynomial_at_zero(budget, spectral_interval)
        certified = tolerance * tolerance
        degree = budget

    final_residual, basis = evaluate_with_cached_products(
        hessian_product, g0, state, coefficients
    )
    final_point = None
    if point is not None:
        correction = correction_chebyshev_coefficients(coefficients, spectral_interval)
        correction_vector = sum(
            float(correction[index]) * basis[index] for index in range(len(correction))
        )
        final_point = np.asarray(point, dtype=float) - correction_vector
    ratio = float(np.linalg.norm(final_residual)) / initial_norm
    return CertifiedSolve(
        residual=final_residual,
        point=final_point,
        state=state,
        design=chosen,
        polynomial_coefficients=coefficients,
        hessian_products=degree,
        chebyshev_budget=budget,
        collective_reductions=1,
        certificate_passed=passed,
        converged=bool(ratio <= tolerance * (1.0 + 2.0e-6)),
        relative_residual=ratio,
        certified_squared_relative_residual=certified,
        design_seconds=elapsed,
    )


def run_five_moment_diagonal(
    *,
    eigenvalues: Array,
    residual: Array,
    spectral_interval: tuple[float, float],
    tolerance: float = 1.0e-4,
    candidate_degrees: Iterable[int] | None = None,
    moment_radii: Array | None = None,
) -> CertifiedSolve:
    """Diagonal reference wrapper used by the reproducible experiments."""

    spectrum = np.asarray(eigenvalues, dtype=float)
    g0 = np.asarray(residual, dtype=float)
    if spectrum.shape != g0.shape or np.any(spectrum <= 0.0):
        raise ValueError("eigenvalues and residual must be aligned and SPD")
    return run_five_moment_method(
        hessian_product=lambda vector: spectrum * vector,
        residual=g0,
        spectral_interval=spectral_interval,
        tolerance=tolerance,
        candidate_degrees=candidate_degrees,
        moment_radii=moment_radii,
    )
