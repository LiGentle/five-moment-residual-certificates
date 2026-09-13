# Online Resource 1: Code and Numerical Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22733853.svg)](https://doi.org/10.5281/zenodo.22733853)

This repository is the public development mirror of the version 1.0.0
reproducibility archive preserved on Zenodo.  The Zenodo record is the
authoritative frozen release for the manuscript.

**Article:** *Polynomial Iteration with Five-Moment Residual Certificates*  
**Journal:** *Journal of Optimization Theory and Applications*  
**Author:** Jintao Li  
**Affiliation:** NUS (Chongqing) Research Institute, Chongqing, China  
**Correspondence:** e0622340@u.nus.edu

This archive contains the implementation, tests, frozen row-level results,
and the boundary-moment, observation-depth, and constrained predictive-control
experiments reported in the paper.

## Environment

The reported computations used Python 3.11.15 on macOS arm64.  Install the
recorded package versions and the local package from the archive root:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Before running the experiments, verify that CVXPY lists `CLARABEL` among its
installed solvers:

```bash
python -c "import cvxpy as cp; assert 'CLARABEL' in cp.installed_solvers()"
```

## Tests

```bash
python -m pytest tests/test_fmcert.py -q
PYTHONPATH=src:experiments python experiments/test_vectorized_sdp.py
PYTHONPATH=src:experiments python -m pytest \
  experiments/test_constrained_mpc.py -q
PYTHONPATH=src:experiments python -m pytest \
  experiments/test_constrained_mpc_jacobi.py -q
```

The expected results are 14 passed tests with no skips and two successful
vectorized-assembly tests, followed by four constrained-control tests.  A solver
may issue an inaccuracy warning on an ill-conditioned test; the returned
polynomial is still subjected to the independent majorant check tested by the
suite.

## Main Validation

```bash
PYTHONPATH=src python experiments/run_validation.py \
  --seed 20260919 --size 384 --degrees 2,3,4,6,8 \
  --tolerances 1e-2,1e-4,1e-6 \
  --output reproduced/independent_end_to_end_v7
```

The runner fails if CLARABEL is unavailable or if the full set of 108 cases is
not constructed.  The frozen output used in the paper is under
`results/independent_end_to_end_v7`.

## Boundary-Moment Diagnostic

```bash
PYTHONPATH=src python experiments/run_atomic_ablation.py \
  --seed 20260919 --size 384 --degrees 2,3,4,6,8 \
  --tolerances 1e-2,1e-4,1e-6 \
  --output reproduced/boundary_moment_diagnostic_v2
```

This second run is a post-development diagnostic, not part of the independent
main validation.  It temporarily disables support recovery within the Python
process and compares the resulting semidefinite computation with the bundled
reference results.  Its frozen output is under
`results/boundary_moment_diagnostic_v2`.

## Observation-Depth Diagnostic

```bash
PYTHONPATH=src:experiments python experiments/run_moment_frontier.py \
  --seed 20260919 --size 384 --tolerance 5e-2 \
  --depths 1,2,3,4,5,6 \
  --output reproduced/moment_frontier_v2
```

The fixed 28-case subset, all accepted degrees, and the independently checked
majorants are recorded under `results/moment_frontier_v2`.  The sparse
vectorized SDP assembly implements the same linear maps as the main code;
`experiments/test_vectorized_sdp.py` compares the two assemblies directly.

## Box-Constrained Predictive-Control Experiment

```bash
PYTHONPATH=src:experiments python experiments/run_constrained_mpc.py \
  --plants double_integrator,damped_oscillator \
  --horizons 20,40,60 --states 8 --state-radius 0.75 \
  --barrier-weights 0.1,0.03,0.01 --tolerance 5e-2 \
  --timing-repeats 5 --seed 20260912 \
  --output reproduced/constrained_mpc_v1
```

This run generates 48 box-constrained problems.  It first replays the Newton
systems on a direct reference path, with every interval-Chebyshev plan
precomputed outside the timed region, and then solves every problem again with
each method following its own barrier-Newton and Armijo path.  Frozen
system-level, timing, and end-to-end data are under
`results/constrained_mpc_v1`.  Timing is single-process Python timing and will
vary by machine; no communication latency is inserted.

## Fixed-Jacobi Sensitivity Check

```bash
PYTHONPATH=src:experiments python \
  experiments/run_constrained_mpc_jacobi_pilot.py \
  --horizons 20,40,60 --barrier-weights 0.1,0.03,0.01 \
  --tolerance 5e-2 --timing-repeats 5 --seed 20260912 \
  --output reproduced/constrained_mpc_jacobi_pilot
```

This exploratory run applies a fixed symmetric Jacobi transformation.  Its
declared stopping test uses the preconditioned residual; the ordinary
Euclidean residual is recorded separately and does not inherit the 0.05
bound.  Full definitions and limitations are recorded in
`results/constrained_mpc_jacobi_pilot/protocol.json`.

`SHA256SUMS` lists the checksum of every other file in this repository.

## Citation

Please cite the archived release:

> Li, J. (2026). *Code and Numerical Data for Polynomial Iteration with
> Five-Moment Residual Certificates* (Version 1.0.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.22733853

Machine-readable citation metadata are provided in `CITATION.cff`.

## License

Python source code is available under the MIT License; see `LICENSE-CODE`.
Numerical data, numerical results, figures, and documentation are available
under the Creative Commons Attribution 4.0 International License; see
`LICENSE-DATA`.  `LICENSE` summarizes the dual-license structure.
