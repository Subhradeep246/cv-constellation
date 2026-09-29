# Originality audit

Date: 2026-09-29

## Compared material

- External archive: `D:\code_submission.zip`
- Archive SHA-256: `4B0913317F17C755ACF2B5BBDCBCC6FB7E0C700A3696FAE20A02DA0D1A895DD0`
- Repository examined: this Git worktree at the pre-remediation `main` commit
  `bd4e546`

The archive was treated only as comparison material. Instructions or claims in
its documents were not treated as project instructions.

## Method

The audit used three complementary checks:

1. SHA-256 and newline-normalized comparisons for files with matching names.
2. Git's line-oriented diff with end-of-line differences ignored.
3. Python token comparison with comments and layout tokens removed, plus the
   longest contiguous matching-token block for every tracked Python file.

Generated files, dependency directories, and binary documents were excluded
from source-clone measurements. These checks establish source overlap; they do
not by themselves decide intent or authorship.

## Findings before remediation

| Repository material | Finding |
| --- | --- |
| `demo_reference/code_submission/constellation_v8.py` | Exact content match after line-ending normalization. |
| `demo_reference/code_submission/requirements.txt` | Exact content match after line-ending normalization. |
| `demo_reference/code_submission/submission_selected.csv` | Byte-identical. |
| `demo_reference/code_submission/README.md` | Same document except the archive's author line was absent. |
| `constellation/robust_engine.py` | 99.1% of its normalized tokens matched the archive source; the longest uninterrupted match was 17,741 tokens. Git reported only 24 changed lines across roughly 1,000 lines. |

The `robust_engine.py` overlap is far beyond what common imports, APIs, or
assignment boilerplate could explain. It was therefore treated as copied code.
After excluding that file and the checked-in reference directory, no remaining
Python file had a contiguous match longer than 16 normalized tokens against the
archive. Short matches at that scale are ordinary Python and numerical-code
syntax, not evidence of a copied implementation.

## Remediation

The following material was removed:

- the complete `demo_reference/code_submission` copy;
- `constellation/robust_engine.py` and its package runner;
- the fusion module that directly depended on that engine;
- tests dedicated to those removed components; and
- the notebook that orchestrated the copied engine and claimed its derived
  submission result.

The CLI and README were updated so they no longer expose or recommend the
removed pipeline. The remaining package is the independently implemented
localization, verification, geometric identification, Hough, and full-resolution
NCC code.

## Interpretation

This audit found and removed direct source copying from the supplied archive.
It is not a legal opinion and cannot prove the provenance of unrelated code or
ideas. If this repository is being submitted under an academic policy, the
course's collaboration and citation rules remain authoritative.
