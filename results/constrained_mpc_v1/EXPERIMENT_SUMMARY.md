# Constrained MPC Newton-system experiment

The paired timing experiment replays time-varying SPD systems collected 
along reference logarithmic-barrier Newton paths. Chebyshev plans are fully prepared 
before timing, and no artificial delay is used.

| Plant | Horizon | Systems | Acceptance | Median cert/Cheb | Mean cert/Cheb | Median gated/Cheb |
|---|---:|---:|---:|---:|---:|---:|
| double_integrator | 20 | 230 | 43.9% | 0.220 | 0.575 | 0.217 |
| double_integrator | 40 | 223 | 35.9% | 0.447 | 0.498 | 0.446 |
| double_integrator | 60 | 217 | 35.9% | 0.516 | 0.468 | 0.517 |
| damped_oscillator | 20 | 232 | 66.8% | 0.204 | 0.071 | 0.200 |
| damped_oscillator | 40 | 226 | 57.5% | 0.273 | 0.118 | 0.240 |
| damped_oscillator | 60 | 226 | 56.2% | 0.289 | 0.111 | 0.232 |

Adaptive CG and direct factorization are reported in `summary.json`; 
the certificate method is not claimed to outperform them. The 
intended comparison is with an unpreconditioned polynomial schedule 
whose coefficients must be fixed before the remaining products are evaluated.

## Independent-path barrier solves

Unlike the paired replay above, this validation advances each method 
along its own barrier-Newton trajectory.

| Method | Converged QPs | Matrix products |
|---|---:|---:|
| Certificate | 48/48 | 510169 |
| Interval Chebyshev | 48/48 | 1200219 |
| CG | 48/48 | 12520 |

The certificate/interval-Chebyshev product ratio is 0.425. 
The maximum relative quadratic-objective gap to the direct reference is 
9.011e-08. 
This is an end-to-end numerical validation of the barrier-QP solves, 
not a closed-loop controller timing comparison.
