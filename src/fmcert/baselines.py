"""Eligible diagonal-model baselines for reproducible spectral experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class IterativeRun:
    method: str
    hessian_products: int
    scalar_reductions: int
    converged: bool
    final_residual_ratio: float


def _inputs(eigenvalues: Array, gradient: Array) -> tuple[Array, Array, float]:
    spectrum = np.asarray(eigenvalues, dtype=float)
    initial = np.asarray(gradient, dtype=float)
    if spectrum.ndim != 1 or spectrum.shape != initial.shape or np.any(spectrum <= 0.0):
        raise ValueError("an aligned positive spectrum and gradient are required")
    norm = float(np.linalg.norm(initial))
    if norm <= 0.0:
        raise ValueError("the initial gradient must be nonzero")
    return spectrum, initial, norm


def run_cg(
    eigenvalues: Array, gradient: Array, *, tolerance: float, max_products: int
) -> IterativeRun:
    spectrum, initial, initial_norm = _inputs(eigenvalues, gradient)
    residual = initial.copy()
    direction = residual.copy()
    residual_squared = float(residual @ residual)
    products = 0
    converged = False
    for iteration in range(1, max_products + 1):
        image = spectrum * direction
        curvature = float(direction @ image)
        if curvature <= 0.0:
            break
        step = residual_squared / curvature
        residual -= step * image
        next_squared = float(residual @ residual)
        products = iteration
        if np.sqrt(next_squared) <= tolerance * initial_norm:
            converged = True
            break
        direction = residual + (next_squared / residual_squared) * direction
        residual_squared = next_squared
    return IterativeRun(
        "cg", products, 2 * products, converged,
        float(np.linalg.norm(residual)) / initial_norm,
    )


def run_minres(
    eigenvalues: Array, gradient: Array, *, tolerance: float, max_products: int
) -> IterativeRun:
    """Paige--Saunders recurrence with externally checked residual norm."""

    spectrum, initial, beta1 = _inputs(eigenvalues, gradient)
    solution = np.zeros_like(initial)
    previous_lanczos = initial.copy()
    current_lanczos = initial.copy()
    old_beta = 0.0
    beta = beta1
    dbar = 0.0
    epsilon = 0.0
    phi_bar = beta1
    cosine = -1.0
    sine = 0.0
    direction = np.zeros_like(initial)
    previous_direction = np.zeros_like(initial)
    products = 0
    for iteration in range(1, max_products + 1):
        vector = current_lanczos / beta
        candidate = spectrum * vector
        products = iteration
        if iteration >= 2:
            candidate -= (beta / old_beta) * previous_lanczos
        alpha = float(vector @ candidate)
        candidate -= (alpha / beta) * current_lanczos
        previous_lanczos, current_lanczos = current_lanczos, candidate
        old_beta = beta
        beta = float(np.linalg.norm(current_lanczos))
        old_epsilon = epsilon
        delta = cosine * dbar + sine * alpha
        gbar = sine * dbar - cosine * alpha
        epsilon = sine * beta
        dbar = -cosine * beta
        gamma = float(np.hypot(gbar, beta))
        if gamma <= np.finfo(float).eps:
            break
        cosine = gbar / gamma
        sine = beta / gamma
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        older_direction = previous_direction
        previous_direction = direction
        direction = (vector - old_epsilon * older_direction - delta * previous_direction) / gamma
        solution += phi * direction
        if abs(phi_bar) <= tolerance * beta1:
            residual = initial - spectrum * solution
            return IterativeRun("minres", products, 2 * products, True, float(np.linalg.norm(residual)) / beta1)
        if beta <= np.finfo(float).eps * beta1:
            break
    residual = initial - spectrum * solution
    return IterativeRun("minres", products, 2 * products, False, float(np.linalg.norm(residual)) / beta1)


def run_bb(
    eigenvalues: Array,
    gradient: Array,
    *,
    rule: str,
    tolerance: float,
    max_products: int,
) -> IterativeRun:
    if rule not in {"bb1", "bb2"}:
        raise ValueError("rule must be bb1 or bb2")
    spectrum, initial, initial_norm = _inputs(eigenvalues, gradient)
    current = initial.copy()
    previous = None
    previous_image = None
    products = 0
    for iteration in range(max_products):
        image = spectrum * current
        products += 1
        if iteration == 0:
            step = float(current @ current) / float(current @ image)
        else:
            assert previous is not None and previous_image is not None
            m0 = float(previous @ previous)
            m1 = float(previous @ previous_image)
            m2 = float(previous_image @ previous_image)
            step = m0 / m1 if rule == "bb1" else m1 / m2
        previous, previous_image = current.copy(), image.copy()
        current = current - step * image
        if float(np.linalg.norm(current)) <= tolerance * initial_norm:
            return IterativeRun(rule, products, products, True, float(np.linalg.norm(current)) / initial_norm)
    return IterativeRun(rule, products, products, False, float(np.linalg.norm(current)) / initial_norm)
