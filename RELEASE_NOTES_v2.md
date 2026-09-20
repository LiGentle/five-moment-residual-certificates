# Version 2 release notes

This release package resolves the provenance and environment inconsistencies
found during the resubmission audit.

## Changes from version 1

- The Table 1 aggregation excludes theoretically exact-zero atomic cases by
  the recorded `atomic_pattern` flag.
- MINRES uses its recurrence residual estimate for the declared stopping test;
  the independently recomputed final residual remains recorded.
- The observation-depth field is named `enclosure_ratio`, not condition number.
- All frozen experiment groups were regenerated or independently verified with
  the exact versions in `requirements-lock.txt`.
- The boundary diagnostic references the main frozen files bundled in this
  same package.
- Every recorded source hash is verified against the actual source file in the
  release package.
- Main figure PDFs and PNGs are linked to their exact CSV inputs by
  `FIGURE_PROVENANCE.json`.
- License files, release verification tools, and a manuscript-alignment report
  are included.

## Published archive

Version 2.0.0 was published on Zenodo on 20 September 2026.  Its
version-specific DOI is 10.5281/zenodo.22857908; the concept DOI for all
versions is 10.5281/zenodo.22733852.  The public Zenodo file MD5 is
37119e179a7ff451da8ce1b87482f2ad.
