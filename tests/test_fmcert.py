from __future__ import annotations

import cvxpy  # required by the semidefinite-design tests
import numpy as np

from fmcert import (
    chebyshev_polynomial_at_zero,
    design_five_moment_polynomial,
    moments_from_spectrum,
    recover_chebyshev_moments,
    recover_five_moment_state,
    run_five_moment_diagonal,
    run_five_moment_method,
    run_minres,
    zero_in_scaled_coordinates,
)


def _state(eigenvalues: np.ndarray, residual: np.ndarray):
    lower = float(np.min(eigenvalues))
    upper = float(np.max(eigenvalues))
    z = (2.0 * eigenvalues - upper - lower) / (upper - lower)
    t1g = z * residual
    t2g = (2.0 * z * z - 1.0) * residual
    return recover_five_moment_state(
        residual,
        t1g,
        t2g,
        spectral_interval=(lower, upper),
    )


def test_two_products_recover_scaled_five_moments() -> None:
    eigenvalues = np.asarray([1.0, 2.0, 7.0, 10.0])
    residual = np.asarray([0.5, -1.0, 0.75, 0.2])
    state = _state(eigenvalues, residual)
    expected = moments_from_spectrum(eigenvalues, residual, (1.0, 10.0))
    np.testing.assert_allclose(state.moments, expected, rtol=3e-13, atol=3e-13)
    assert state.duplicate_second_moment_discrepancy < 1.0e-13


def test_s_probes_recover_two_s_plus_one_moments() -> None:
    rng = np.random.default_rng(314)
    eigenvalues = np.exp(rng.uniform(0.0, np.log(80.0), 96))
    eigenvalues[[0, -1]] = [1.0, 80.0]
    residual = rng.normal(size=96)
    z = (2.0 * eigenvalues - 81.0) / 79.0
    basis = np.polynomial.chebyshev.chebvander(z, 3).T * residual
    recovered = recover_chebyshev_moments(basis)
    weights = residual * residual / float(residual @ residual)
    expected = weights @ np.polynomial.chebyshev.chebvander(z, 6)
    np.testing.assert_allclose(recovered, expected, rtol=5e-13, atol=5e-13)


def test_state_is_invariant_to_common_matrix_scaling() -> None:
    rng = np.random.default_rng(12)
    eigenvalues = np.exp(rng.uniform(0.0, np.log(50.0), 80))
    eigenvalues[[0, -1]] = [1.0, 50.0]
    residual = rng.normal(size=80)
    first = _state(eigenvalues, residual)
    z = (2.0 * eigenvalues - 51.0) / 49.0
    scaled = recover_five_moment_state(
        residual,
        z * residual,
        (2.0 * z * z - 1.0) * residual,
        spectral_interval=(17.0, 850.0),
    )
    np.testing.assert_allclose(first.moments, scaled.moments, rtol=3e-12, atol=3e-12)


def test_degree_two_value_is_exact_and_matches_two_step_minres() -> None:
    rng = np.random.default_rng(41)
    eigenvalues = np.exp(rng.uniform(0.0, np.log(30.0), 100))
    eigenvalues[[0, -1]] = [1.0, 30.0]
    residual = rng.normal(size=100)
    state = _state(eigenvalues, residual)
    design = design_five_moment_polynomial(state.moments, 2, (1.0, 30.0))
    z = (2.0 * eigenvalues - 31.0) / 29.0
    actual = float(np.sum(residual**2 * design.values(z) ** 2) / (residual @ residual))
    np.testing.assert_allclose(design.certificate, actual, rtol=3e-10, atol=3e-12)
    minres = run_minres(eigenvalues, residual, tolerance=1.0e-15, max_products=2)
    np.testing.assert_allclose(
        np.sqrt(actual), minres.final_residual_ratio, rtol=2e-9, atol=2e-11
    )


def test_two_atom_problem_is_certified_in_two_products() -> None:
    eigenvalues = np.r_[np.ones(40), np.full(60, 100.0)]
    residual = np.random.default_rng(42).normal(size=eigenvalues.size)
    run = run_five_moment_diagonal(
        eigenvalues=eigenvalues,
        residual=residual,
        spectral_interval=(1.0, 100.0),
        tolerance=1.0e-5,
        candidate_degrees=(2,),
    )
    assert run.certificate_passed
    assert run.hessian_products == 2
    assert run.hessian_products < run.chebyshev_budget
    assert run.relative_residual < 1.0e-10
    assert run.converged


def test_one_active_atom_uses_the_two_product_probe_without_crashing() -> None:
    eigenvalues = np.full(48, 5.0)
    residual = np.random.default_rng(420).normal(size=eigenvalues.size)
    run = run_five_moment_diagonal(
        eigenvalues=eigenvalues,
        residual=residual,
        spectral_interval=(1.0, 10.0),
        tolerance=1.0e-6,
        candidate_degrees=(2,),
    )
    assert run.certificate_passed
    assert run.hessian_products == 2
    assert len(run.polynomial_coefficients) == 3
    assert run.relative_residual < 1.0e-12
    assert run.converged


def test_endpoint_three_atom_problem_is_certified_in_three_products() -> None:
    eigenvalues = np.r_[np.ones(40), np.full(30, 8.0), np.full(30, 40.0)]
    residual = np.random.default_rng(9).normal(size=eigenvalues.size)
    run = run_five_moment_diagonal(
        eigenvalues=eigenvalues,
        residual=residual,
        spectral_interval=(1.0, 40.0),
        tolerance=1.0e-5,
        candidate_degrees=(2, 3),
    )
    assert run.certificate_passed
    assert run.hessian_products == 3
    assert run.design is not None and run.design.atomic_pattern
    assert run.relative_residual < 1.0e-9


def test_fallback_reproduces_the_chebyshev_budget() -> None:
    rng = np.random.default_rng(4)
    eigenvalues = np.exp(rng.uniform(0.0, np.log(100.0), 160))
    eigenvalues[[0, -1]] = [1.0, 100.0]
    residual = rng.normal(size=160)
    run = run_five_moment_diagonal(
        eigenvalues=eigenvalues,
        residual=residual,
        spectral_interval=(1.0, 100.0),
        tolerance=1.0e-8,
        candidate_degrees=(2,),
    )
    assert not run.certificate_passed
    assert run.hessian_products == run.chebyshev_budget
    assert run.converged
    baseline = chebyshev_polynomial_at_zero(run.chebyshev_budget, (1.0, 100.0))
    np.testing.assert_allclose(run.polynomial_coefficients, baseline)


def test_loose_target_skips_the_two_product_probe() -> None:
    eigenvalues = np.asarray([1.0, 1.25, 1.5, 2.0])
    residual = np.asarray([1.0, -0.3, 0.5, 0.2])
    run = run_five_moment_diagonal(
        eigenvalues=eigenvalues,
        residual=residual,
        spectral_interval=(1.0, 2.0),
        tolerance=0.5,
    )
    assert run.chebyshev_budget == 1
    assert run.hessian_products == 1
    assert run.collective_reductions == 0
    assert run.state is None
    assert run.converged


def test_returned_point_has_the_reported_residual() -> None:
    eigenvalues = np.r_[np.ones(12), np.full(9, 7.0), np.full(11, 30.0)]
    rng = np.random.default_rng(881)
    point = rng.normal(size=eigenvalues.size)
    residual = rng.normal(size=eigenvalues.size)
    right_hand_side = eigenvalues * point - residual
    run = run_five_moment_method(
        hessian_product=lambda vector: eigenvalues * vector,
        residual=residual,
        point=point,
        spectral_interval=(1.0, 30.0),
        tolerance=1.0e-5,
        candidate_degrees=(2, 3),
    )
    assert run.point is not None
    np.testing.assert_allclose(
        eigenvalues * run.point - right_hand_side,
        run.residual,
        rtol=2e-10,
        atol=2e-10,
    )


def test_sdp_certificate_covers_realized_residual_and_improves_chebyshev_bound() -> None:
    rng = np.random.default_rng(19)
    eigenvalues = np.r_[np.exp(rng.uniform(0.0, np.log(5.0), 127)), 100.0]
    eigenvalues[0] = 1.0
    residual = rng.normal(size=128)
    eta = moments_from_spectrum(eigenvalues, residual, (1.0, 100.0))
    design = design_five_moment_polynomial(eta, 4, (1.0, 100.0))
    z = (2.0 * eigenvalues - 101.0) / 99.0
    actual = float(np.sum(residual**2 * design.values(z) ** 2) / (residual @ residual))
    assert actual <= design.certificate + 2.0e-7
    z0 = zero_in_scaled_coordinates((1.0, 100.0))
    chebyshev_bound = 1.0 / float(np.cosh(4 * np.arccosh(-z0))) ** 2
    assert design.certificate < chebyshev_bound


def test_interval_moment_certificate_covers_a_perturbed_state() -> None:
    rng = np.random.default_rng(29)
    eigenvalues = np.exp(rng.uniform(0.0, np.log(20.0), 120))
    eigenvalues[[0, -1]] = [1.0, 20.0]
    residual = rng.normal(size=120)
    eta = moments_from_spectrum(eigenvalues, residual, (1.0, 20.0))
    nominal = design_five_moment_polynomial(eta, 4, (1.0, 20.0))
    perturbation = np.asarray([0.0, 2e-6, -3e-6, 1e-6, -2e-6])
    estimate = eta + perturbation
    radii = np.abs(perturbation) + 1.0e-8
    radii[0] = 0.0
    robust = design_five_moment_polynomial(
        estimate, 4, (1.0, 20.0), moment_radii=radii
    )
    z = (2.0 * eigenvalues - 21.0) / 19.0
    actual = float(np.sum(residual**2 * robust.values(z) ** 2) / (residual @ residual))
    assert actual <= robust.certificate + 2.0e-7
    assert robust.certificate >= nominal.design_value - 1.0e-6
    assert robust.design_value >= nominal.design_value - 1.0e-6


def test_robust_extraction_handles_a_nearly_atomic_moment_matrix() -> None:
    eigenvalues = np.r_[np.ones(60), np.full(30, np.sqrt(1000.0)), np.full(30, 1000.0)]
    residual = np.random.default_rng(2026).normal(size=eigenvalues.size)
    eta = moments_from_spectrum(eigenvalues, residual, (1.0, 1000.0))
    radii = np.asarray([0.0, 2e-6, 2e-6, 3e-6, 3e-6])
    estimate = eta + 0.75 * np.asarray([0.0, 1.0, -1.0, 1.0, -1.0]) * radii
    robust = design_five_moment_polynomial(
        estimate, 4, (1.0, 1000.0), moment_radii=radii
    )
    z = (2.0 * eigenvalues - 1001.0) / 999.0
    actual = float(np.sum(residual**2 * robust.values(z) ** 2) / (residual @ residual))
    assert actual <= robust.certificate + 2.0e-7
    assert robust.certificate <= 1.25 * robust.design_value + 2.0e-6


def test_rank_one_lmi_extraction_avoids_ill_conditioned_inverse() -> None:
    estimate = np.asarray(
        [
            1.0,
            0.9978247351268328,
            0.9997320951366458,
            0.9983256299621849,
            0.9990587849481597,
        ]
    )
    radii = np.asarray([0.0, 2.0e-6, 2.0e-6, 3.0e-6, 3.0e-6])
    robust = design_five_moment_polynomial(
        estimate, 4, (1.0, 1000.0), moment_radii=radii
    )
    assert "rank-one-lmi-nullspace" in robust.status
    assert robust.certificate <= 1.1 * robust.design_value + 2.0e-6
