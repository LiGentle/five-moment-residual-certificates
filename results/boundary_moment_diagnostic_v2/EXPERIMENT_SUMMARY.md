# Boundary-moment implementation diagnostic

> **Status.** Post-development diagnostic only; this is not part of the
> independent reference validation.

The experiment disables boundary-moment recognition only within the running
Python process and redesigns all 60 synthetic reference cases on degrees
`{2,3,4,6,8}`.  No core source file is modified.

| target | reference accepted | shortcut off | retained acceptances |
|---:|---:|---:|---:|
| 1e-02 | 36/60 | 34/60 | 34/36 (94.4%) |
| 1e-04 | 20/60 | 8/60 | 8/20 (40.0%) |
| 1e-06 | 8/60 | 1/60 | 1/8 (12.5%) |

Of 300 design attempts, 296 returned and 4 failed.  All failures were singular degree-two solves in the two-point
family.  Returned designs had zero certificate violations, and the
Chebyshev fallback left all target runs valid and converged.

All 12 early acceptances outside the two- and three-point families at
tolerance `1e-2` were unchanged.  Every lost acceptance came from the
two- or three-point families.

Across those two families, the reference run used 0.102 s for the candidate grids, whereas the ablation used 86.55 s (850 times as long).

The result separates safety from numerical sharpness: the certificate
and fallback remain safe without the shortcut, but reliable tight-target
termination on singular boundary moment sequences depends strongly on
explicit treatment of boundary moment sequences.
