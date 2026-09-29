# Constellation detection

A CPU implementation for the CS-GY 6643 constellation-detection competition.
It discovers scene files, locates query patches in 3000 x 3000 sky images,
ranks the supplied constellation diagrams, and writes competition-format CSVs.
Inference uses the supplied images and pattern drawings; it does not contain
scene-specific answers.

The repository was audited against a separately supplied student submission.
An almost verbatim engine and a checked-in copy of that submission were removed.
See [ORIGINALITY.md](ORIGINALITY.md) for the reproducible findings and remediation
scope. The commands documented below use the independently implemented
`constellation` pipeline only.

## Setup

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records the versions used for local benchmarks. The
dataset is intentionally ignored by Git. By default, the CLI expects this
layout under `data/constellation-detection/participant`:

```text
participant/
├── patterns/
├── train/
├── validation/
├── train_ground_truth.csv
└── sample_submission.csv
```

Pass `--data C:\path\to\participant` to use another location.

## Main commands

```powershell
# Inspect data quality, duplicates, and preprocessing ablations.
.\.venv\Scripts\python.exe -m constellation audit --output outputs/patch-audit

# Evaluate on the labelled training scenes.
.\.venv\Scripts\python.exe -m constellation evaluate --output outputs/evaluation

# Generate and validate a complete validation submission.
.\.venv\Scripts\python.exe -m constellation predict --output outputs/prediction

# Run the automated checks.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`predict` writes `submission.csv` only after all requested scenes pass schema,
name, tuple, patch-count, and coordinate-bound validation. A partial scene run
is deliberately disallowed for submission prediction.

## Method

1. Detect source candidates with a difference of Gaussians and Harris corners.
2. Use RootSIFT correspondences and RANSAC to estimate patch pose when enough
   distinctive features are available.
3. Fall back to a rotation-invariant polar Fourier descriptor, then refine the
   shortlisted positions with photometric verification.
4. Extract star nodes from the supplied RGBA constellation drawings and rank
   affine point-set hypotheses with outlier rejection.
5. Emit `(x, y, membership)` for present patches and `-1` otherwise.

The default path is intentionally conservative. Extra modes below are
experiments and should be evaluated separately:

```powershell
# Denser position search.
.\.venv\Scripts\python.exe -m constellation evaluate --grid-step 4 --top-k 100 --output outputs/dense

# Copy-aware global pose search.
.\.venv\Scripts\python.exe -m constellation evaluate --hough-identification --output outputs/hough

# Broad-star support plus full-resolution masked-NCC corrections.
.\.venv\Scripts\python.exe -m constellation evaluate --hough-identification --bright-star-weight 1 --fullres-ncc --output outputs/fullres

# Keep full-resolution candidates in a reusable, content-checked cache.
.\.venv\Scripts\python.exe -m constellation predict --hough-identification --bright-star-weight 1 --fullres-ncc --fullres-cache-dir outputs/fullres-cache --output outputs/fullres-validation
```

The full-resolution candidate reported a 0.62054 public score, but a public
leaderboard result is not a substitute for held-out validation. Detailed local
measurements and failed ablations are retained in [BENCHMARKS.md](BENCHMARKS.md).

## Diagnostics

```powershell
# Small labelled smoke test; its score is marked partial.
.\.venv\Scripts\python.exe -m constellation evaluate --scene taurus --limit-patches 8 --skip-identification --output outputs/smoke

# Evaluate constellation ranking from oracle training coordinates.
.\.venv\Scripts\python.exe -m constellation oracle-identify --output outputs/oracle

# Probe detector coverage and verification at oracle locations.
.\.venv\Scripts\python.exe -m constellation.diagnose

# Select thresholds on two scenes and report the held-out third scene.
.\.venv\Scripts\python.exe -m constellation.calibrate outputs/evaluation/evaluate-patches.json --output outputs/calibration.json
```

See [CLEANING.md](CLEANING.md) for preprocessing and duplicate-image analysis,
[PLAN.md](PLAN.md) for the development plan, and [BENCHMARKS.md](BENCHMARKS.md)
for measurements.

## Limits

- Repetitive or degraded patches can produce confident matches at the wrong
  location.
- The candidate detector can miss true centres; denser search improves recall
  at a substantial CPU cost.
- Affine point-set ranking can prefer chance alignments in clutter, especially
  for small diagrams.
- Only three labelled scenes are available locally, so threshold tuning and
  ablations have high variance.
- The local metric follows the supplied textual formula; edge-case conventions
  may differ from the competition server.

No GPU libraries or credentials are required after the dataset is available.
