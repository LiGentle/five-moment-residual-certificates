# Test report

- Date: 20 September 2026
- Interpreter: CPython 3.11.15
- Environment: exact `requirements-lock.txt` versions
- Source under test: this release directory (`PYTHONPATH=src`)

## Results

| suite | result |
|---|---:|
| `tests/test_fmcert.py` | 14 passed |
| `experiments/test_vectorized_sdp.py` | 2 passed |
| constrained MPC and fixed-Jacobi tests | 4 passed |

The core suite emitted two CVXPY warnings that a solution may be inaccurate in
ill-conditioned cases.  Both cases passed the independent certificate checks;
there were no failures or skipped tests.

The release verifier subsequently checked the pinned scientific summaries,
all embedded source/result/reference hashes, figure/data provenance, and every
file in `SHA256SUMS`.
