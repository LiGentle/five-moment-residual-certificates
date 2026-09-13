# Fixed-Jacobi constrained-MPC pilot

This exploratory check uses the fixed symmetric transformation
`P = diag(H)^(-1/2)`.  The declared stopping test is
`||P(Ap+g)||_2 / ||Pg||_2 <= 0.05`.  The ordinary residual
`||Ap+g||_2 / ||g||_2` is recorded separately and is not covered by that
tolerance.  Complete definitions and timing details are in `protocol.json`.

All six replay configurations favor the certificate method over the
precomputed interval-Chebyshev schedule in both median and mean kernel time.

| Plant | Horizon | Systems | Acceptance | Median time ratio | Mean time ratio | Mean products (cert/Cheb) |
|---|---:|---:|---:|---:|---:|---:|
| double integrator | 20 | 230 | 35.2% | 0.438 | 0.598 | 209.1 / 361.6 |
| double integrator | 40 | 223 | 28.7% | 0.560 | 0.602 | 352.7 / 601.4 |
| double integrator | 60 | 217 | 23.5% | 0.748 | 0.650 | 510.1 / 797.6 |
| damped oscillator | 20 | 232 | 63.4% | 0.289 | 0.090 | 17.2 / 312.0 |
| damped oscillator | 40 | 226 | 55.8% | 0.280 | 0.123 | 28.4 / 313.0 |
| damped oscillator | 60 | 226 | 54.9% | 0.302 | 0.124 | 30.2 / 327.6 |

Across the 1,354 replay systems, the weighted mean kernel-time ratio is
0.442 and the matrix-product ratio is 0.419.  Adaptive CG remains much
faster.

In the independent-path barrier solves, all methods converge on all 48 QPs
with no box-constraint violation.  The certificate method uses 468,442
matrix products, compared with 763,370 for interval Chebyshev and 16,282 for
CG.  The certificate/Chebyshev ratio is 0.614.  The six per-configuration
ratios are 0.595, 0.720, 0.993, 0.113, 0.112, and 0.193; hence the result at
double-integrator horizon 60 has little margin.  The maximum relative
objective gap from the direct reference is 1.69e-7.

The largest ordinary relative residual is 0.190 for the certificate method,
0.085 for interval Chebyshev, and 0.901 for CG.  This does not contradict the
declared preconditioned-residual test.  It limits the conclusion: the pilot
supports robustness of the fixed-schedule comparison under one fixed Jacobi
transformation, but it does not establish a 0.05 bound in the ordinary
Euclidean residual and is not evidence of superiority over CG.

Reproduction command:

```bash
env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:experiments \
  /opt/homebrew/anaconda3/envs/spdppp/bin/python \
  experiments/run_constrained_mpc_jacobi_pilot.py \
  --output results/constrained_mpc_jacobi_pilot
```
