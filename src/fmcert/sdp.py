"""Exact small-SDP design for five-moment residual polynomials.

The design problem maximizes the Christoffel value over all moment extensions
compatible with the observed first five Chebyshev moments.  A second SDP
constructs a polynomial majorant of the selected squared residual.  The latter
is the deployable certificate; a final interval correction covers the conic
solver tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebmul, chebval

from .moments import chebyshev_to_power_moments

Array = np.ndarray


@dataclass(frozen=True)
class MomentPolynomial:
    """A residual polynomial and its sharp five-moment upper bound."""

    degree: int
    coefficients: Array
    certificate: float
    design_value: float
    dual_majorant: Array
    status: str
    solver: str
    atomic_pattern: bool
    atomic_nodes: Array | None
    atomic_weights: Array | None
    interval_correction: float

    def values(self, scaled_eigenvalues: Array) -> Array:
        return np.asarray(chebval(scaled_eigenvalues, self.coefficients), dtype=float)


def zero_in_scaled_coordinates(spectral_interval: tuple[float, float]) -> float:
    lower, upper = (float(value) for value in spectral_interval)
    if not (0.0 < lower < upper):
        raise ValueError("spectral_interval must satisfy 0 < lower < upper")
    return -(upper + lower) / (upper - lower)


def _normalize_residual(coefficients: Array, z0: float) -> Array:
    """Return coefficients numerically normalized by p(z0)=1."""

    normalized = np.asarray(coefficients, dtype=float).copy()
    value = float(chebval(z0, normalized))
    if not np.isfinite(value) or abs(value) <= np.finfo(float).tiny:
        raise RuntimeError("the residual polynomial cannot be normalized at zero")
    normalized /= value
    # A second constant-term correction removes the last evaluation roundoff.
    normalized[0] += 1.0 - float(chebval(z0, normalized))
    return normalized


def chebyshev_polynomial_at_zero(
    degree: int, spectral_interval: tuple[float, float]
) -> Array:
    """Return the degree-``degree`` interval Chebyshev residual coefficients."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    z0 = zero_in_scaled_coordinates(spectral_interval)
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0 / float(chebval(z0, np.eye(degree + 1)[degree]))
    return _normalize_residual(coefficients, z0)


def _chebyshev_gram_numeric(moments: Array, degree: int) -> Array:
    eta = np.asarray(moments, dtype=float)
    if eta.size < 2 * degree + 1:
        raise ValueError("insufficient moments for the requested Gram matrix")
    return np.asarray(
        [
            [0.5 * (eta[i + j] + eta[abs(i - j)]) for j in range(degree + 1)]
            for i in range(degree + 1)
        ],
        dtype=float,
    )


def _recover_atomic_measure(
    chebyshev_moments: Array, *, tolerance: float = 5.0e-12
) -> tuple[Array, Array] | None:
    """Recognize determinate one/two-atom and endpoint-three-atom sequences."""

    y = chebyshev_to_power_moments(chebyshev_moments)
    moment_matrix = np.asarray([[y[i + j] for j in range(3)] for i in range(3)])
    scale = max(1.0, float(np.linalg.norm(moment_matrix, ord=2)))
    eigenvalues = np.linalg.eigvalsh(moment_matrix)
    rank = int(np.sum(eigenvalues > tolerance * scale))
    candidates: list[Array] = []
    if rank == 1:
        candidates.append(np.asarray([y[1] / y[0]]))
    elif rank == 2:
        hankel = np.asarray([[y[0], y[1]], [y[1], y[2]]])
        if np.linalg.cond(hankel) < 1.0 / tolerance:
            constant, linear = np.linalg.solve(hankel, -np.asarray([y[2], y[3]]))
            roots = np.roots([1.0, linear, constant])
            if np.max(np.abs(np.imag(roots))) <= 10.0 * tolerance:
                candidates.append(np.sort(np.real(roots)))

    localizing = np.asarray(
        [[y[0] - y[2], y[1] - y[3]], [y[1] - y[3], y[2] - y[4]]]
    )
    local_scale = max(1.0, float(np.linalg.norm(localizing, ord=2)))
    local_eigenvalues, local_vectors = np.linalg.eigh(localizing)
    local_rank = int(np.sum(local_eigenvalues > tolerance * local_scale))
    if local_rank == 0:
        candidates.append(np.asarray([-1.0, 1.0]))
    elif local_rank == 1 and local_eigenvalues[0] <= tolerance * local_scale:
        null = local_vectors[:, 0]
        if abs(null[1]) > tolerance:
            interior = -null[0] / null[1]
            if -1.0 + tolerance < interior < 1.0 - tolerance:
                candidates.append(np.asarray([-1.0, interior, 1.0]))

    for nodes in candidates:
        vandermonde = np.vander(nodes, N=len(nodes), increasing=True).T
        try:
            weights = np.linalg.solve(vandermonde, y[: len(nodes)])
        except np.linalg.LinAlgError:
            continue
        reconstructed = np.asarray(
            [float(weights @ nodes**order) for order in range(5)], dtype=float
        )
        error = float(np.max(np.abs(reconstructed - y)))
        if (
            np.all(nodes >= -1.0 - tolerance)
            and np.all(nodes <= 1.0 + tolerance)
            and np.all(weights >= -tolerance)
            and error <= 50.0 * tolerance * max(1.0, float(np.max(np.abs(y))))
        ):
            return np.clip(nodes, -1.0, 1.0), np.maximum(weights, 0.0)
    return None


def _cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as error:  # pragma: no cover - exercised in clean envs
        raise RuntimeError(
            "CVXPY is required for degrees above two; install requirements.txt"
        ) from error
    return cp


def _gram_expression(cp: Any, moments: Any, degree: int) -> Any:
    return cp.bmat(
        [
            [
                0.5 * (moments[i + j] + moments[abs(i - j)])
                for j in range(degree + 1)
            ]
            for i in range(degree + 1)
        ]
    )


def _localizing_expression(cp: Any, moments: Any, degree: int) -> Any:
    """Return ``integral (1-z^2) T_i T_j`` for ``i,j<degree``."""

    weight = np.asarray([0.5, 0.0, -0.5])
    rows = []
    for i in range(degree):
        left = np.zeros(i + 1)
        left[i] = 1.0
        row = []
        for j in range(degree):
            right = np.zeros(j + 1)
            right[j] = 1.0
            coefficients = chebmul(chebmul(left, right), weight)
            row.append(
                sum(float(value) * moments[index] for index, value in enumerate(coefficients))
            )
        rows.append(row)
    return cp.bmat(rows)


def _gram_polynomial_coefficients(matrix: Any, degree: int) -> list[Any]:
    """Chebyshev coefficients of ``T(z)^T matrix T(z)``."""

    coefficients: list[Any] = [0 for _ in range(2 * degree + 1)]
    for i in range(degree + 1):
        for j in range(degree + 1):
            coefficients[i + j] = coefficients[i + j] + 0.5 * matrix[i, j]
            coefficients[abs(i - j)] = coefficients[abs(i - j)] + 0.5 * matrix[i, j]
    return coefficients


def _multiplication_matrix(input_size: int, multiplier: Array) -> Array:
    output_size = input_size + len(multiplier) - 1
    matrix = np.zeros((output_size, input_size))
    for index in range(input_size):
        basis = np.zeros(index + 1)
        basis[index] = 1.0
        product = chebmul(basis, multiplier)
        matrix[: len(product), index] = product
    return matrix


def _solve(problem: Any, cp: Any, preferred_solver: str) -> tuple[str, str]:
    installed = set(cp.installed_solvers())
    choices = [preferred_solver, "CLARABEL", "SCS"]
    last_error: Exception | None = None
    for solver in dict.fromkeys(choices):
        if solver not in installed:
            continue
        try:
            if solver == "CLARABEL":
                problem.solve(
                    solver=solver,
                    tol_gap_abs=1.0e-9,
                    tol_gap_rel=1.0e-9,
                    tol_feas=1.0e-9,
                    max_iter=1000,
                )
            elif solver == "SCS":
                problem.solve(solver=solver, eps=2.0e-7, max_iters=200000)
            else:
                problem.solve(solver=solver)
        except Exception as error:  # pragma: no cover - solver-specific
            last_error = error
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return str(problem.status), solver
    raise RuntimeError(f"no conic solver returned a solution: {last_error}")


def _maximum_chebyshev_polynomial(coefficients: Array) -> float:
    polynomial = Chebyshev(np.asarray(coefficients, dtype=float))
    candidates = [-1.0, 1.0]
    for root in polynomial.deriv().roots():
        if abs(float(np.imag(root))) <= 1.0e-9 and -1.0 <= float(np.real(root)) <= 1.0:
            candidates.append(float(np.real(root)))
    return float(np.max(polynomial(candidates)))


def certify_fixed_polynomial(
    moments: Array,
    polynomial: Array,
    *,
    moment_radii: Array | None = None,
    preferred_solver: str = "CLARABEL",
) -> tuple[float, Array, float, str, str]:
    """Compute an SOS upper bound for a fixed squared residual polynomial."""

    cp = _cvxpy()
    eta = np.asarray(moments, dtype=float)
    residual = np.trim_zeros(np.asarray(polynomial, dtype=float), trim="b")
    degree = len(residual) - 1
    if degree < 1:
        raise ValueError("the residual polynomial must have positive degree")
    radii = np.zeros_like(eta) if moment_radii is None else np.asarray(moment_radii, dtype=float)
    if eta.ndim != 1 or eta.size < 1 or radii.shape != eta.shape or np.any(radii < 0.0):
        raise ValueError("moments and moment_radii must be aligned vectors")

    q = cp.Variable(eta.size)
    q0 = cp.Variable((degree + 1, degree + 1), PSD=True)
    q1 = cp.Variable((degree, degree), PSD=True)
    sigma0 = _gram_polynomial_coefficients(q0, degree)
    sigma1 = _gram_polynomial_coefficients(q1, degree - 1)
    multiplier = _multiplication_matrix(len(sigma1), np.asarray([0.5, 0.0, -0.5]))
    localized = [
        sum(multiplier[row, col] * sigma1[col] for col in range(len(sigma1)))
        for row in range(2 * degree + 1)
    ]
    squared = chebmul(residual, residual)
    constraints = []
    for index in range(2 * degree + 1):
        q_coefficient = q[index] if index < eta.size else 0.0
        constraints.append(q_coefficient - squared[index] == sigma0[index] + localized[index])
    objective = eta @ q + radii @ cp.abs(q)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    status, solver = _solve(problem, cp, preferred_solver)
    dual = np.asarray(q.value, dtype=float)

    padded = np.pad(dual, (0, max(0, len(squared) - len(dual))))
    violation_polynomial = squared - padded[: len(squared)]
    violation = _maximum_chebyshev_polynomial(violation_polynomial)
    arithmetic = 5.0e-9 * (1.0 + float(np.sum(np.abs(dual))) + float(np.sum(np.abs(squared))))
    correction = max(0.0, violation) + arithmetic
    dual[0] += correction
    bound = float(dual @ eta + np.abs(dual) @ radii)
    return max(0.0, bound), dual, correction, status, solver


def design_moment_polynomial(
    moments: Array,
    degree: int,
    spectral_interval: tuple[float, float],
    *,
    moment_radii: Array | None = None,
    preferred_solver: str = "CLARABEL",
) -> MomentPolynomial:
    """Solve the conditional minimax design for any supplied moment prefix.

    ``moments[j]`` is the expectation of ``T_j`` on ``[-1,1]``.  This generic
    routine is used for the one-, three-, and five-moment information ablation.
    The specialized five-moment wrapper additionally recognizes determinate
    finite spectra and has a closed-form degree-two branch.
    """

    eta = np.asarray(moments, dtype=float)
    if eta.ndim != 1 or eta.size < 1 or not np.isclose(eta[0], 1.0, atol=1.0e-8):
        raise ValueError("moments must be a normalized Chebyshev-moment prefix")
    if degree < 1 or eta.size > 2 * degree + 1:
        raise ValueError("degree is incompatible with the supplied moment prefix")
    radii = (
        np.zeros_like(eta)
        if moment_radii is None
        else np.asarray(moment_radii, dtype=float)
    )
    if radii.shape != eta.shape or np.any(radii < 0.0) or radii[0] != 0.0:
        raise ValueError("moment_radii must be nonnegative, aligned, and exact in mass")
    cp = _cvxpy()
    z0 = zero_in_scaled_coordinates(spectral_interval)
    extension = cp.Variable(2 * degree + 1)
    gamma = cp.Variable()
    evaluation = np.asarray(
        [chebval(z0, np.eye(degree + 1)[index]) for index in range(degree + 1)],
        dtype=float,
    )
    gram = _gram_expression(cp, extension, degree)
    constraints = [
        gamma >= 0.0,
        gram - gamma * np.outer(evaluation, evaluation) >> 0,
        _localizing_expression(cp, extension, degree) >> 0,
    ]
    if np.any(radii > 0.0):
        constraints.extend(
            [extension[: eta.size] >= eta - radii, extension[: eta.size] <= eta + radii]
        )
    else:
        constraints.append(extension[: eta.size] == eta)
    problem = cp.Problem(cp.Maximize(gamma), constraints)
    status, design_solver = _solve(problem, cp, preferred_solver)
    extended = np.asarray(extension.value, dtype=float)
    numerical_gram = _chebyshev_gram_numeric(extended, degree)
    numerical_gram = 0.5 * (numerical_gram + numerical_gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(numerical_gram)
    scale = max(1.0, float(eigenvalues[-1]))
    positive_value = float(problem.value) > 1.0e-12 * scale
    candidate_polynomials: list[tuple[str, Array]] = []
    if not positive_value:
        nullspace = eigenvectors[:, eigenvalues <= 2.0e-7 * scale]
        projected = nullspace @ (nullspace.T @ evaluation)
        if abs(float(evaluation @ projected)) <= 1.0e-12:
            raise RuntimeError("failed to extract a zero-value saddle polynomial")
        candidate_polynomials.append(
            ("moment-nullspace", projected / float(evaluation @ projected))
        )
    else:
        # At an exact positive optimum, the minimizer spans the nullspace of
        # G-gamma vv^T.  Extracting that null vector is markedly more stable
        # than explicitly inverting an ill-conditioned moment matrix.
        lmi = numerical_gram - float(problem.value) * np.outer(evaluation, evaluation)
        lmi = 0.5 * (lmi + lmi.T)
        lmi_values, lmi_vectors = np.linalg.eigh(lmi)
        null_candidate = None
        for index in np.argsort(np.abs(lmi_values)):
            candidate = lmi_vectors[:, index]
            candidate_normalization = float(evaluation @ candidate)
            if abs(candidate_normalization) > 1.0e-12 * float(
                np.linalg.norm(evaluation)
            ):
                null_candidate = candidate / candidate_normalization
                break
        if null_candidate is None:
            raise RuntimeError("failed to extract the positive-value saddle polynomial")
        candidate_polynomials.append(("rank-one-lmi-nullspace", null_candidate))

        # An inaccurate primal extension may have several nearly null
        # directions.  The leading vector of the rank-one LMI dual is an
        # independent saddle candidate; certify both and keep the safer one.
        if "inaccurate" in status and constraints[1].dual_value is not None:
            dual_matrix = np.asarray(constraints[1].dual_value, dtype=float)
            dual_matrix = 0.5 * (dual_matrix + dual_matrix.T)
            dual_values, dual_vectors = np.linalg.eigh(dual_matrix)
            for index in np.argsort(dual_values)[::-1]:
                candidate = dual_vectors[:, index]
                candidate_normalization = float(evaluation @ candidate)
                if abs(candidate_normalization) > 1.0e-12 * float(
                    np.linalg.norm(evaluation)
                ):
                    candidate_polynomials.append(
                        ("rank-one-lmi-dual", candidate / candidate_normalization)
                    )
                    break

    certified_candidates: list[
        tuple[float, str, Array, Array, float, str, str]
    ] = []
    for extraction_method, candidate in candidate_polynomials:
        residual = _normalize_residual(candidate, z0)
        try:
            bound, dual, correction, certificate_status, certificate_solver = (
                certify_fixed_polynomial(
                    eta,
                    residual,
                    moment_radii=radii,
                    preferred_solver=preferred_solver,
                )
            )
        except RuntimeError:
            continue
        certified_candidates.append(
            (
                bound,
                extraction_method,
                residual,
                dual,
                correction,
                certificate_status,
                certificate_solver,
            )
        )
    if not certified_candidates:
        raise RuntimeError("failed to certify every extracted saddle polynomial")
    (
        bound,
        extraction_method,
        residual,
        dual,
        correction,
        certificate_status,
        certificate_solver,
    ) = min(certified_candidates, key=lambda item: item[0])
    return MomentPolynomial(
        degree=degree,
        coefficients=np.asarray(residual, dtype=float),
        certificate=bound,
        design_value=max(0.0, float(problem.value)),
        dual_majorant=dual,
        status=(
            f"design={status}; certificate={certificate_status}; "
            f"extraction={extraction_method}; candidates={len(certified_candidates)}"
        ),
        solver=f"{design_solver}/{certificate_solver}",
        atomic_pattern=False,
        atomic_nodes=None,
        atomic_weights=None,
        interval_correction=correction,
    )


def _atomic_residual(nodes: Array, z0: float) -> Array:
    polynomial = Polynomial([1.0])
    for node in nodes:
        polynomial *= Polynomial([-float(node), 1.0])
    polynomial /= float(polynomial(z0))
    converted = polynomial.convert(kind=Chebyshev)
    return _normalize_residual(np.asarray(converted.coef, dtype=float), z0)


def _atomic_majorant(nodes: Array, residual: Array, z0: float) -> Array:
    """Return a degree-four majorant for an atomic candidate.

    For one or two proposed nodes the squared residual itself has degree at
    most four.  For the endpoint--interior--endpoint pattern, cancel one
    factor ``1-z^2`` from the square; the remaining nonnegative degree-four
    polynomial still majorizes the square on ``[-1,1]``.  Thus this bound is
    valid even if a floating-point rank test proposed the wrong nodes.
    """

    if len(nodes) <= 2:
        majorant = chebmul(residual, residual)
    elif len(nodes) == 3 and abs(float(nodes[0]) + 1.0) <= 1.0e-7 and abs(float(nodes[-1]) - 1.0) <= 1.0e-7:
        middle = float(nodes[1])
        denominator = float(np.prod(z0 - nodes))
        power = Polynomial([1.0, 0.0, -1.0]) * Polynomial([-middle, 1.0]) ** 2
        power /= denominator * denominator
        majorant = np.asarray(power.convert(kind=Chebyshev).coef, dtype=float)
    else:  # Defensive fallback; current recovery never proposes this pattern.
        raise RuntimeError("no analytic degree-four majorant for proposed atoms")
    squared = chebmul(residual, residual)
    padded = np.pad(majorant, (0, max(0, len(squared) - len(majorant))))
    violation = _maximum_chebyshev_polynomial(squared - padded[: len(squared)])
    correction = max(0.0, violation) + 128.0 * np.finfo(float).eps * (
        1.0 + float(np.sum(np.abs(squared))) + float(np.sum(np.abs(majorant)))
    )
    majorant = np.pad(majorant, (0, max(0, 5 - len(majorant))))[:5]
    majorant[0] += correction
    return majorant


def design_five_moment_polynomial(
    moments: Array,
    degree: int,
    spectral_interval: tuple[float, float],
    *,
    moment_radii: Array | None = None,
    preferred_solver: str = "CLARABEL",
) -> MomentPolynomial:
    """Solve the five-moment conditional minimax problem at one degree."""

    eta = np.asarray(moments, dtype=float)
    if eta.shape != (5,) or not np.isclose(eta[0], 1.0, atol=1.0e-8):
        raise ValueError("moments must be normalized and have length five")
    if degree < 2:
        raise ValueError("degree must be at least two")
    radii = (
        np.zeros(5)
        if moment_radii is None
        else np.asarray(moment_radii, dtype=float)
    )
    if radii.shape != (5,) or np.any(radii < 0.0) or radii[0] != 0.0:
        raise ValueError("moment_radii must have length five and exact mass")
    z0 = zero_in_scaled_coordinates(spectral_interval)
    atomic = _recover_atomic_measure(eta)
    if not np.any(radii > 0.0) and atomic is not None and len(atomic[0]) <= degree:
        nodes, weights = atomic
        residual = _atomic_residual(nodes, z0)
        dual = _atomic_majorant(nodes, residual, z0)
        value = max(0.0, float(dual @ eta + np.abs(dual) @ radii))
        # The probe has already used two HVPs. A one-atom measure produces a
        # linear annihilator, but the executed horizon is still two; padding
        # makes that accounting explicit and permits cached-basis evaluation.
        executed_degree = max(2, len(residual) - 1)
        residual = np.pad(residual, (0, executed_degree + 1 - len(residual)))
        correction = 0.0
        status = "solver-corrected candidate from a numerical atomic pattern"
        solver = "analytic"
        return MomentPolynomial(
            degree=executed_degree,
            coefficients=residual,
            certificate=value,
            design_value=float("nan"),
            dual_majorant=dual,
            status=status,
            solver=solver,
            atomic_pattern=True,
            atomic_nodes=nodes,
            atomic_weights=weights,
            interval_correction=correction,
        )

    if degree == 2 and not np.any(radii > 0.0):
        gram = _chebyshev_gram_numeric(eta, 2)
        evaluation = np.asarray(
            [chebval(z0, np.eye(3)[index]) for index in range(3)], dtype=float
        )
        inverse_evaluation = np.linalg.solve(gram, evaluation)
        residual = inverse_evaluation / float(evaluation @ inverse_evaluation)
        residual = _normalize_residual(residual, z0)
        squared = chebmul(residual, residual)
        # The first five moments evaluate a quadratic residual exactly in
        # exact arithmetic. The coefficient norm can nevertheless amplify
        # floating-point errors in the recovered moments. Add a transparent
        # scale-aware margin, analogous to the fixed-polynomial correction.
        arithmetic = 128.0 * np.finfo(float).eps * (
            1.0 + float(np.sum(np.abs(squared)))
        )
        dual = np.asarray(squared, dtype=float).copy()
        dual[0] += arithmetic
        value = float(dual @ eta + np.abs(dual) @ radii)
        return MomentPolynomial(
            degree=2,
            coefficients=residual,
            certificate=max(0.0, value),
            design_value=max(0.0, float(squared @ eta)),
            dual_majorant=dual,
            status="exact degree-two moment calculation",
            solver="linear algebra",
            atomic_pattern=False,
            atomic_nodes=None,
            atomic_weights=None,
            interval_correction=arithmetic,
        )

    return design_moment_polynomial(
        eta,
        degree,
        spectral_interval,
        moment_radii=radii,
        preferred_solver=preferred_solver,
    )
