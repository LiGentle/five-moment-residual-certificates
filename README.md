# Code and Numerical Data for Polynomial Iteration with Five-Moment Residual Certificates

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22857908.svg)](https://doi.org/10.5281/zenodo.22857908)

- **Article:** *Polynomial Iteration with Five-Moment Residual Certificates*
- **Journal:** *Journal of Optimization Theory and Applications*
- **Author:** Jintao Li
- **Version:** 2.0.0
- **Version DOI:** https://doi.org/10.5281/zenodo.22857908
- **Concept DOI (all versions):** https://doi.org/10.5281/zenodo.22733852

This repository is the public development mirror of Online Resource 1 version
2.0.0.  The authoritative frozen release is preserved on Zenodo under the
version-specific DOI above.  The version 2.0.0 files, provenance metadata, and
numerical snapshot in this repository correspond to that archived release.

## Frozen environment

The v2 frozen computations used Python 3.11.15 on macOS arm64 and the exact
package versions in `requirements-lock.txt`.  In particular, the numerical
path used NumPy 2.4.6, CVXPY 1.9.2, Clarabel 0.11.1, SCS 3.2.11,
scikit-learn 1.9.0, and Matplotlib 3.11.0.

Create an isolated environment and install the package from the archive root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
```

## Verification

Verify the release metadata, frozen scientific summaries, embedded source
hashes, figure/data links, and whole-archive checksums with:

```bash
python tools/verify_release.py
shasum -a 256 -c SHA256SUMS
```

Run the test suite with:

```bash
python -m pytest tests/test_fmcert.py -q
PYTHONPATH=src:experiments python experiments/test_vectorized_sdp.py
PYTHONPATH=src:experiments python -m pytest \
  experiments/test_constrained_mpc.py \
  experiments/test_constrained_mpc_jacobi.py -q
```

The frozen release audit obtained 14 passing core tests, two passing
vectorized-assembly tests, and four passing constrained-control tests.  Two
CVXPY inaccuracy warnings occur in ill-conditioned certificate tests; the
independent majorant checks still pass.

## Frozen experiments

The following directories were generated from the source files included here,
under the pinned environment above:

- `results/independent_end_to_end_v7`: 108-case main validation, 540
  fixed-degree checks, 324 target runs, and 36 robust-moment checks;
- `results/moment_frontier_v2`: the prespecified 28-case observation-depth
  diagnostic;
- `results/boundary_moment_diagnostic_v2`: the 60-case post-development
  boundary-moment ablation, using the frozen main run in this package as its
  reference;
- `results/constrained_mpc_v1`: the 1,354-system box-constrained predictive-
  control replay and the 48-problem end-to-end comparison;
- `results/constrained_mpc_jacobi_pilot`: the fixed-Jacobi sensitivity check.

Each applicable JSON summary records `source_sha256`.  Those hashes are checked
against the source files in this package by `tools/verify_release.py`; they were
not rewritten to impersonate a different generating source.

The main figures in both PDF and PNG form are stored beside the main CSV files.
`FIGURE_PROVENANCE.json` links each figure to the exact frozen CSV and generator
hash.  They use enlarged, embedded TrueType text for legibility in the 12-point
referee manuscript.  The CSV files remain the authoritative data source.

## Reproduction commands

Main validation:

```bash
PYTHONPATH=src python experiments/run_validation.py \
  --seed 20260919 --size 384 --degrees 2,3,4,6,8 \
  --tolerances 1e-2,1e-4,1e-6 \
  --output reproduced/independent_end_to_end_v7
```

Submission figures from the frozen main-validation CSV files:

```bash
python experiments/render_submission_figures.py \
  --snapshot results/independent_end_to_end_v7 \
  --output reproduced/submission_figures
```

Boundary-moment diagnostic (run after the main validation is present under
`results/independent_end_to_end_v7`):

```bash
PYTHONPATH=src python experiments/run_atomic_ablation.py \
  --seed 20260919 --size 384 --degrees 2,3,4,6,8 \
  --tolerances 1e-2,1e-4,1e-6 \
  --output reproduced/boundary_moment_diagnostic_v2
```

Observation-depth diagnostic:

```bash
PYTHONPATH=src:experiments python experiments/run_moment_frontier.py \
  --seed 20260919 --size 384 --tolerance 5e-2 \
  --depths 1,2,3,4,5,6 \
  --output reproduced/moment_frontier_v2
```

Predictive-control replay:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src:experiments python experiments/run_constrained_mpc.py \
  --plants double_integrator,damped_oscillator \
  --horizons 20,40,60 --states 8 --state-radius 0.75 \
  --barrier-weights 0.1,0.03,0.01 --tolerance 5e-2 \
  --timing-repeats 5 --seed 20260912 \
  --output reproduced/constrained_mpc_v1
```

Fixed-Jacobi sensitivity check:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src:experiments python \
  experiments/run_constrained_mpc_jacobi_pilot.py \
  --horizons 20,40,60 --barrier-weights 0.1,0.03,0.01 \
  --tolerance 5e-2 --timing-repeats 5 --seed 20260912 \
  --output reproduced/constrained_mpc_jacobi_pilot
```

## Timing policy

Every timing value in this package is the frozen observation from one local
run.  Wall-clock and microsecond measurements are machine-, load-, library-,
and run-dependent; exact timing reproduction is neither expected nor used as a
correctness criterion.  Counts, acceptance decisions, residual checks, and
coverage results are the platform-independent scientific comparisons.

The accompanying JOTA revision quotes this same frozen snapshot.
`CONSISTENCY_REPORT.md` lists the exact values and records the completed
manuscript-alignment check.

## Licenses and citation

Python source code is licensed under the MIT License (`LICENSE-CODE`).  Data,
results, figures, and documentation are licensed under CC BY 4.0
(`LICENSE-DATA`).  `CITATION.cff` records version 2.0.0 and its version-specific
DOI.  To identify the exact reproducibility archive, cite:

> Li, J. (2026). *Code and Numerical Data for Polynomial Iteration with
> Five-Moment Residual Certificates* (Version 2.0.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.22857908
