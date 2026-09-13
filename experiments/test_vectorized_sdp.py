"""Equivalence checks for the vectorized experimental SDP assembly."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander

from fmcert import design_moment_polynomial
from fmcert.sdp import _chebyshev_gram_numeric
from run_validation import build_cases
from vectorized_sdp import (
    _gram_map,
    _localizing_map,
    design_vectorized_moment_polynomial,
)


class VectorizedSdpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = next(
            case
            for case in build_cases(20260919, 384)
            if case.name == "poisson2d_n144_0"
        )
        lower, upper = cls.case.interval
        cls.scaled = (
            2.0 * cls.case.eigenvalues - upper - lower
        ) / (upper - lower)
        cls.weights = cls.case.residual**2 / float(
            cls.case.residual @ cls.case.residual
        )

    def test_linear_maps_match_direct_formulas(self) -> None:
        degree = 8
        moments = self.weights @ chebvander(self.scaled, 2 * degree)
        mapped_gram = np.asarray(_gram_map(degree) @ moments).reshape(
            degree + 1, degree + 1
        )
        np.testing.assert_allclose(
            mapped_gram,
            _chebyshev_gram_numeric(moments, degree),
            rtol=0.0,
            atol=2.0e-15,
        )
        mapped_localizing = np.asarray(_localizing_map(degree) @ moments).reshape(
            degree, degree
        )
        # Independent quadrature over the realized finite spectral measure.
        vandermonde = chebvander(self.scaled, degree - 1)
        direct = (vandermonde.T * (self.weights * (1.0 - self.scaled**2))) @ vandermonde
        np.testing.assert_allclose(mapped_localizing, direct, rtol=0.0, atol=2.0e-14)

    def test_design_and_majorant_match_reference_assembly(self) -> None:
        for depth, degree in ((1, 4), (1, 8), (2, 4), (2, 8), (3, 4), (3, 8), (4, 4), (4, 8)):
            with self.subTest(depth=depth, degree=degree):
                moments = self.weights @ chebvander(self.scaled, 2 * depth)
                reference = design_moment_polynomial(
                    moments, degree, self.case.interval
                )
                vectorized = design_vectorized_moment_polynomial(
                    moments, degree, self.case.interval
                )
                self.assertAlmostEqual(
                    reference.design_value, vectorized.design_value, delta=2.0e-10
                )
                self.assertAlmostEqual(
                    reference.certificate, vectorized.certificate, delta=2.0e-8
                )
                values = chebval(self.scaled, vectorized.coefficients)
                actual = float(self.weights @ values**2)
                self.assertLessEqual(
                    actual,
                    vectorized.certificate
                    + 1024.0
                    * np.finfo(float).eps
                    * (1.0 + vectorized.certificate + np.sum(abs(vectorized.coefficients)) ** 2),
                )


if __name__ == "__main__":
    unittest.main()
