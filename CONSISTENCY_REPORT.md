# Online Resource 1 v2 consistency report

## Verdict

The scientific outputs in this release package form one pinned-environment
snapshot, and their embedded source hashes match the included generating
source.  The manuscript values reported below use the same snapshot.

## Main validation

The five Table 1 rows are:

| degree | median certificate/Chebyshev | strict improvement | median certificate/actual, non-atomic |
|---:|---:|---:|---:|
| 2 | 0.0641 | 100.0% | 1.00 |
| 3 | 0.0422 | 100.0% | 1.49 |
| 4 | 0.0476 | 100.0% | 2.18 |
| 6 | 0.0553 | 100.0% | 1.88 |
| 8 | 0.0588 | 100.0% | 1.85 |

There were zero violations in 540 fixed-degree checks, 324 target runs, and 36
robust-moment checks.  The frozen candidate-grid timing was median 0.5702 s,
maximum 8.9690 s; these values are timing observations, not correctness claims.

The main PDF/PNG figures were rendered deterministically from the exact bundled
`design_results.csv` and `target_results.csv` files by the included plotting
script.  Their data, generator, and output hashes are recorded in
`FIGURE_PROVENANCE.json`.

## Observation-depth and boundary diagnostics

The depth-1 through depth-6 median product counts are 11.0, 9.5, 8.5, 7.0,
5.0, and 6.0.  All 168 decisions are covered.  The frozen total design time was
61.0297 s.

With boundary handling disabled, 296 of 300 designs returned and four singular
degree-two cases failed.  Early acceptances were 34, 8, and 1 at targets
1e-2, 1e-4, and 1e-6.  The two-/three-point candidate-grid totals in this
snapshot were 0.0551 s for the reference and 33.9788 s for the ablation.

## Predictive-control experiment

All non-timing counts agree with the manuscript's current scientific claims:

- 1,354 reference-path Newton systems;
- end-to-end certificate/Chebyshev updates: 1,409/1,426;
- end-to-end certificate/Chebyshev products: 510,169/1,200,219;
- maximum certificate relative residual: 0.0498627;
- largest certificate objective gap to direct: 9.011e-8;
- fixed-Jacobi certificate product reduction: 38.6350%.

The v2 frozen replay timings are:

| plant | horizon | median us, cert/Cheb | mean us, cert/Cheb |
|---|---:|---:|---:|
| integrator | 20 | 108.875/495.520 | 1306.572/2272.894 |
| integrator | 40 | 296.042/662.083 | 1466.485/2947.005 |
| integrator | 60 | 497.583/965.041 | 1453.195/3106.403 |
| oscillator | 20 | 35.312/172.895 | 78.471/1109.791 |
| oscillator | 40 | 48.021/175.813 | 121.907/1035.938 |
| oscillator | 60 | 57.417/198.604 | 140.155/1265.570 |

Across all replay rows, the frozen median/mean times were 72.125/753.554 us
for the certificate and 422.458/1943.611 us for interval Chebyshev.  The
certificate had lower mean total replay time on 42 of 48 paths.  Frozen median
CG/direct replay times were 16.625/16.167 us.

## Manuscript alignment

The JOTA revision dated 20 September 2026 matches this snapshot in its counts,
residual checks, coverage results, Table 1 entries, observation-depth results,
boundary diagnostic, constrained-control table, and timing values.  The text
identifies every wall-clock value as a frozen local observation rather than a
correctness criterion.  In particular, it uses 0.5702/8.9690 s for the main
candidate grid, 0.0551/33.9788 s for the boundary comparison, the replay values
listed above, and 42/48 path wins.

## Automated checks

`tools/verify_release.py` verifies all embedded source hashes, result/reference
hashes, figure-data hashes, the pinned scientific summaries, and the complete
`SHA256SUMS` manifest.  The release test run produced 14 passing core tests,
two passing vectorized tests, and four passing constrained-control tests.
