"""Stable recovery of five spectral moments from a Chebyshev probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FiveMomentState:
    """Normalized Chebyshev moments and cached Krylov vectors."""

    moments: Array
    chebyshev_gradient: Array
    second_chebyshev_gradient: Array
    spectral_interval: tuple[float, float]
    initial_squared_residual_norm: float
    minimum_hausdorff_eigenvalue: float
    duplicate_second_moment_discrepancy: float


def recover_chebyshev_moments(basis_vectors: Array) -> Array:
    """Recover moments through order ``2s`` from ``T_0(Z)g,...,T_s(Z)g``.

    The rows of ``basis_vectors`` are the cached Chebyshev vectors.  All
    required information is contained in their Gram matrix and can therefore
    be accumulated by one fused reduction in a distributed implementation.
    """

    basis = np.asarray(basis_vectors, dtype=float)
    if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
        raise ValueError("basis_vectors must be a nonempty two-dimensional array")
    if np.any(~np.isfinite(basis)):
        raise ValueError("basis_vectors must be finite")
    gram = basis @ basis.T
    mass = float(gram[0, 0])
    if mass <= 0.0:
        raise ValueError("the initial residual must be nonzero")
    depth = basis.shape[0] - 1
    moments = np.empty(2 * depth + 1, dtype=float)
    moments[0] = 1.0
    for order in range(1, depth + 1):
        moments[order] = float(gram[0, order]) / mass
    for order in range(depth + 1, 2 * depth + 1):
        estimates = []
        for left in range(max(0, order - depth), min(depth, order) + 1):
            right = order - left
            if left > right or right > depth:
                continue
            estimates.append(
                2.0 * float(gram[left, right]) / mass - moments[abs(left - right)]
            )
        if not estimates:
            raise RuntimeError("internal moment-recovery failure")
        moments[order] = float(np.mean(estimates))
    return moments


def chebyshev_to_power_moments(moments: Array) -> Array:
    """Convert moments of ``T_0,...,T_4`` to moments of ``1,z,...,z^4``."""

    eta = np.asarray(moments, dtype=float)
    if eta.shape != (5,):
        raise ValueError("moments must have length five")
    power = np.empty(5, dtype=float)
    power[0] = eta[0]
    power[1] = eta[1]
    power[2] = 0.5 * (eta[2] + eta[0])
    power[3] = 0.25 * (eta[3] + 3.0 * eta[1])
    power[4] = 0.125 * (eta[4] + 8.0 * power[2] - eta[0])
    return power


def _minimum_hausdorff_eigenvalue(power_moments: Array) -> float:
    """Return the smallest order-four Hausdorff LMI eigenvalue on [-1,1]."""

    y = np.asarray(power_moments, dtype=float)
    moment_matrix = np.asarray([[y[i + j] for j in range(3)] for i in range(3)])
    localizing = np.asarray(
        [[y[i + j] - y[i + j + 2] for j in range(2)] for i in range(2)]
    )
    return float(
        min(
            np.min(np.linalg.eigvalsh(moment_matrix)),
            np.min(np.linalg.eigvalsh(localizing)),
        )
    )


def recover_five_moment_state(
    residual: Array,
    chebyshev_residual: Array,
    second_chebyshev_residual: Array,
    *,
    spectral_interval: tuple[float, float],
    feasibility_tolerance: float = 1.0e-8,
) -> FiveMomentState:
    """Recover five scale-invariant moments from ``T_0(Z)g,T_1(Z)g,T_2(Z)g``.

    Here ``Z=(2H-LI-mu I)/(L-mu)``.  These vectors should be generated directly
    by the Chebyshev recurrence.  Doing so avoids the cancellation in first
    forming ``H^2g`` and then shifting and scaling it.  The two nontrivial
    vectors are cached and reused when the selected residual polynomial is
    evaluated.  The six Gram entries can be combined in one reduction.
    """

    vectors = [
        np.asarray(value, dtype=float)
        for value in (residual, chebyshev_residual, second_chebyshev_residual)
    ]
    if vectors[0].ndim != 1 or any(value.shape != vectors[0].shape for value in vectors):
        raise ValueError("T_0(Z)g, T_1(Z)g, and T_2(Z)g must be aligned vectors")
    if any(np.any(~np.isfinite(value)) for value in vectors):
        raise ValueError("Krylov vectors must be finite")
    lower, upper = (float(value) for value in spectral_interval)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("spectral_interval must satisfy 0 < lower < upper")

    g0, t1g, t2g = vectors
    norm2 = float(g0 @ g0)
    if norm2 <= 0.0:
        raise ValueError("the initial residual must be nonzero")
    eta1 = float(g0 @ t1g) / norm2
    eta2_direct = float(g0 @ t2g) / norm2
    eta2_gram = 2.0 * float(t1g @ t1g) / norm2 - 1.0
    eta2 = 0.5 * (eta2_direct + eta2_gram)
    eta3 = 2.0 * float(t1g @ t2g) / norm2 - eta1
    eta4 = 2.0 * float(t2g @ t2g) / norm2 - 1.0
    moments = np.asarray([1.0, eta1, eta2, eta3, eta4], dtype=float)
    if np.any(~np.isfinite(moments)):
        raise ValueError("the recovered moments are not finite")

    minimum_eigenvalue = _minimum_hausdorff_eigenvalue(
        chebyshev_to_power_moments(moments)
    )
    if minimum_eigenvalue < -abs(float(feasibility_tolerance)):
        raise ValueError(
            "recovered moments are incompatible with the spectral interval: "
            f"minimum LMI eigenvalue {minimum_eigenvalue:.3e}"
        )
    return FiveMomentState(
        moments=moments,
        chebyshev_gradient=t1g,
        second_chebyshev_gradient=t2g,
        spectral_interval=(lower, upper),
        initial_squared_residual_norm=norm2,
        minimum_hausdorff_eigenvalue=minimum_eigenvalue,
        duplicate_second_moment_discrepancy=abs(eta2_direct - eta2_gram),
    )


def moments_from_spectrum(
    eigenvalues: Array,
    residual: Array,
    spectral_interval: tuple[float, float],
) -> Array:
    """Reference moment computation used only by tests and spectral studies."""

    values = np.asarray(eigenvalues, dtype=float)
    vector = np.asarray(residual, dtype=float)
    if values.shape != vector.shape or np.any(values <= 0.0):
        raise ValueError("eigenvalues and residual must be aligned and positive")
    lower, upper = spectral_interval
    z = (2.0 * values - upper - lower) / (upper - lower)
    weights = vector * vector / float(vector @ vector)
    vandermonde = np.polynomial.chebyshev.chebvander(z, 4)
    return np.asarray(weights @ vandermonde, dtype=float)
