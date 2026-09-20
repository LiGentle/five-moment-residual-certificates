# Observation-depth diagnostic

The case subset and tolerance were specified by the script before the run. The horizon is the first degree whose independently checked majorant reaches the target; it includes the products used to acquire the moments.

- Cases: 28
- Target relative residual: 0.05
- Certificate violations: 0
- Total local SDP time: 61.03 s

| Initial products s | Median horizon | Median horizon / Chebyshev | Strictly below Chebyshev | Median saving when improved |
|---:|---:|---:|---:|---:|
| 1 | 11.0 | 0.818 | 20/28 | 20.5% |
| 2 | 9.5 | 0.750 | 25/28 | 26.3% |
| 3 | 8.5 | 0.662 | 27/28 | 36.4% |
| 4 | 7.0 | 0.608 | 28/28 | 39.2% |
| 5 | 5.0 | 0.500 | 28/28 | 50.0% |
| 6 | 6.0 | 0.600 | 28/28 | 40.0% |

The median horizon is smallest at s=5.  It rises at s=6 even though the fixed-degree bound cannot increase, because the observation itself then costs six products.  Thus taking more moments has a measurable but nonmonotone net value.
