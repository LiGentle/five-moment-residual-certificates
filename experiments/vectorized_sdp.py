"""Vectorized CVXPY assembly used by the observation-depth diagnostic.

The production implementation deliberately favors formulas that mirror the
proof.  At degrees near twenty, however, constructing thousands of scalar
CVXPY expressions dominates the solve.  This module assembles the identical
linear maps as sparse matrices.  It is kept in ``experiments`` because it is a
timing/diagnostic implementation, not a second mathematical method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.polynomial.chebyshev import chebmul, chebval
from scipy import sparse

from fmcert.sdp import (
    _maximum_chebyshev_polynomial,
    _normalize_residual,
    _solve,
    zero_in_scaled_coordinates,
)


@dataclass(frozen=True)
class VectorizedMomentPolynomial:
    degree: int
    coefficients: np.ndarray
    certificate: float
    design_value: float
    status: str
    solver: str


def _gram_map(degree: int) -> sparse.csr_matrix:
    width = 2 * degree + 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            row = i * (degree + 1) + j
            rows.extend((row, row))
            cols.extend((i + j, abs(i - j)))
            data.extend((0.5, 0.5))
    return sparse.coo_matrix((data, (rows, cols)), shape=((degree + 1) ** 2, width)).tocsr()


def _localizing_map(degree: int) -> sparse.csr_matrix:
    width = 2 * degree + 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    weight = np.asarray([0.5, 0.0, -0.5])
    for i in range(degree):
        left = np.zeros(i + 1)
        left[i] = 1.0
        for j in range(degree):
            right = np.zeros(j + 1)
            right[j] = 1.0
            coefficients = chebmul(chebmul(left, right), weight)
            row = i * degree + j
            for index, value in enumerate(coefficients):
                if value:
                    rows.append(row)
                    cols.append(index)
                    data.append(float(value))
    return sparse.coo_matrix((data, (rows, cols)), shape=(degree**2, width)).tocsr()


def _quadratic_form_map(degree: int) -> sparse.csr_matrix:
    """Map row-major vec(Q) to coefficients of T(z)^T Q T(z)."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            column = i * (degree + 1) + j
            rows.extend((i + j, abs(i - j)))
            cols.extend((column, column))
            data.extend((0.5, 0.5))
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(2 * degree + 1, (degree + 1) ** 2)
    ).tocsr()


def _multiplication_map(input_size: int, multiplier: np.ndarray) -> sparse.csr_matrix:
    output_size = input_size + len(multiplier) - 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for index in range(input_size):
        basis = np.zeros(index + 1)
        basis[index] = 1.0
        product = chebmul(basis, multiplier)
        for row, value in enumerate(product):
            if value:
                rows.append(row)
                cols.append(index)
                data.append(float(value))
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(output_size, input_size)
    ).tocsr()


def _numeric_gram(moments: np.ndarray, degree: int) -> np.ndarray:
    mapping = _gram_map(degree)
    return np.asarray(mapping @ moments).reshape((degree + 1, degree + 1))


def _certificate(
    moments: np.ndarray, polynomial: np.ndarray, preferred_solver: str
) -> tuple[float, str, str]:
    eta = np.asarray(moments, dtype=float)
    residual = np.trim_zeros(np.asarray(polynomial, dtype=float), trim="b")
    degree = len(residual) - 1
    q = cp.Variable(eta.size)
    q0 = cp.Variable((degree + 1, degree + 1), PSD=True)
    q1 = cp.Variable((degree, degree), PSD=True)
    q0_map = _quadratic_form_map(degree)
    q1_map = _quadratic_form_map(degree - 1)
    multiplier = _multiplication_map(2 * degree - 1, np.asarray([0.5, 0.0, -0.5]))
    q_coefficients = cp.hstack([q, np.zeros(2 * degree + 1 - eta.size)])
    right = (
        q0_map @ cp.reshape(q0, ((degree + 1) ** 2,), order="C")
        + multiplier @ q1_map @ cp.reshape(q1, (degree**2,), order="C")
    )
    squared = chebmul(residual, residual)
    problem = cp.Problem(cp.Minimize(eta @ q), [q_coefficients - squared == right])
    status, solver = _solve(problem, cp, preferred_solver)
    dual = np.asarray(q.value, dtype=float)
    padded = np.pad(dual, (0, max(0, len(squared) - len(dual))))
    violation = _maximum_chebyshev_polynomial(squared - padded[: len(squared)])
    arithmetic = 5.0e-9 * (
        1.0 + float(np.sum(np.abs(dual))) + float(np.sum(np.abs(squared)))
    )
    dual[0] += max(0.0, violation) + arithmetic
    return max(0.0, float(dual @ eta)), status, solver


def design_vectorized_moment_polynomial(
    moments: np.ndarray,
    degree: int,
    spectral_interval: tuple[float, float],
    *,
    preferred_solver: str = "CLARABEL",
) -> VectorizedMomentPolynomial:
    eta = np.asarray(moments, dtype=float)
    if eta.ndim != 1 or eta.size < 1 or not np.isclose(eta[0], 1.0, atol=1.0e-8):
        raise ValueError("moments must be a normalized Chebyshev-moment prefix")
    if degree < 1 or eta.size > 2 * degree + 1:
        raise ValueError("degree is incompatible with the supplied moment prefix")
    z0 = zero_in_scaled_coordinates(spectral_interval)
    extension = cp.Variable(2 * degree + 1)
    gamma = cp.Variable()
    evaluation = np.asarray([chebval(z0, [0.0] * i + [1.0]) for i in range(degree + 1)])
    gram = cp.reshape(_gram_map(degree) @ extension, (degree + 1, degree + 1), order="C")
    localizing = cp.reshape(
        _localizing_map(degree) @ extension, (degree, degree), order="C"
    )
    constraints = [
        gamma >= 0.0,
        gram - gamma * np.outer(evaluation, evaluation) >> 0,
        localizing >> 0,
        extension[: eta.size] == eta,
    ]
    problem = cp.Problem(cp.Maximize(gamma), constraints)
    status, design_solver = _solve(problem, cp, preferred_solver)
    extended = np.asarray(extension.value, dtype=float)
    numerical_gram = _numeric_gram(extended, degree)
    numerical_gram = 0.5 * (numerical_gram + numerical_gram.T)
    lmi = numerical_gram - float(problem.value) * np.outer(evaluation, evaluation)
    lmi = 0.5 * (lmi + lmi.T)
    values, vectors = np.linalg.eigh(lmi)
    candidates: list[np.ndarray] = []
    for index in np.argsort(np.abs(values)):
        trial = vectors[:, index]
        normalization = float(evaluation @ trial)
        if abs(normalization) > 1.0e-12 * float(np.linalg.norm(evaluation)):
            candidates.append(trial / normalization)
            break
    if not candidates:
        # This branch occurs only at an essentially zero design value.  The
        # nullspace of the moment matrix then contains an annihilator.
        values, vectors = np.linalg.eigh(numerical_gram)
        scale = max(1.0, float(values[-1]))
        nullspace = vectors[:, values <= 2.0e-7 * scale]
        projected = nullspace @ (nullspace.T @ evaluation)
        normalization = float(evaluation @ projected)
        if abs(normalization) <= 1.0e-12:
            raise RuntimeError("failed to extract a saddle polynomial")
        candidates.append(projected / normalization)
    if "inaccurate" in status and constraints[1].dual_value is not None:
        dual = np.asarray(constraints[1].dual_value, dtype=float)
        dual = 0.5 * (dual + dual.T)
        dual_values, dual_vectors = np.linalg.eigh(dual)
        for index in np.argsort(dual_values)[::-1]:
            trial = dual_vectors[:, index]
            normalization = float(evaluation @ trial)
            if abs(normalization) > 1.0e-12 * float(np.linalg.norm(evaluation)):
                candidates.append(trial / normalization)
                break
    certified: list[tuple[float, np.ndarray, str, str]] = []
    for candidate in candidates:
        residual = _normalize_residual(candidate, z0)
        try:
            certificate, certificate_status, certificate_solver = _certificate(
                eta, residual, preferred_solver
            )
        except RuntimeError:
            continue
        certified.append(
            (certificate, residual, certificate_status, certificate_solver)
        )
    if not certified:
        raise RuntimeError("failed to certify every extracted saddle polynomial")
    certificate, residual, certificate_status, certificate_solver = min(
        certified, key=lambda item: item[0]
    )
    return VectorizedMomentPolynomial(
        degree=degree,
        coefficients=residual,
        certificate=certificate,
        design_value=max(0.0, float(problem.value)),
        status=f"design={status}; certificate={certificate_status}",
        solver=f"{design_solver}/{certificate_solver}",
    )
