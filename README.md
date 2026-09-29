# Constellation detection

## Final submission (public score 0.94)

Submit [`outputs/final/submission.csv`](outputs/final/submission.csv). The
reproducible notebook is [`final_constellation_pipeline.ipynb`](final_constellation_pipeline.ipynb):
open it from this repository with the `.venv` Python kernel and run all cells.
It regenerates the CSV from validation images, using the locally cached
matching evidence in `outputs/final/cache` when available. The first uncached
run can take hours. The notebook's temporary intermediate CSVs are removed
after the final output is validated. No selected reference CSV, training labels,
external catalogue, or synthetic images are used during validation inference.

The final method conservatively fuses equal-weight and raw-heavy image matches.
The standalone raw-heavy CSV scored 0.92 and is **not** the final submission.
The fused CSV scored **0.94**, as reported by the user. Historical experiments
below are retained for context, but their unscored output folders were cleaned
up. The retained output folders are `final`, `demo-v8-selected`,
`hough-validation`, `hough-broadstar-validation`, and
`hough-fullres-patchonly-validation`.

For the latest non-destructive cleaning, copied-neighborhood audit, and full
preprocessing ablation results, see [CLEANING.md](CLEANING.md). Additional filters,
including two tested starlet-denoising strengths, remain disabled by default
because each reduced the three-scene mean score.

A runnable CPU implementation for the supplied CS-GY 6643 competition. It processes actual image pixels, discovers scene files, localizes patches, ranks reference constellations, and writes a competition-format CSV. No scene-specific answers are built into inference. GPU libraries are not required.

## Run on this device

The project already has a configured `.venv` (Python 3.12.14). From this project folder in PowerShell:

```powershell
# Inspect the dataset, patch quality, and exact vs similar-looking duplicates.
# Original images are not rewritten; duplicate patch IDs are kept.
.\.venv\Scripts\python.exe -m constellation audit --output outputs/patch-audit

# Fast evaluation on all three labelled scenes.
.\.venv\Scripts\python.exe -m constellation evaluate --output outputs/my-evaluation

# More exhaustive position search; slower, with bounded working memory.
.\.venv\Scripts\python.exe -m constellation evaluate --grid-step 4 --top-k 100 --output outputs/my-thorough-evaluation

# Predict all validation scenes and write a local submission CSV.
.\.venv\Scripts\python.exe -m constellation predict --output outputs/my-predictions

# Run mathematical, image-registration and CSV-contract checks.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The matcher supports `--background-sigma` for cloud and background subtraction.
Values 4, 10, 14, 20, and 28 all scored below the default 7 on the three labelled
scenes; use other values only for controlled experiments.

For optional wavelet denoising experiments, use `--preprocess starlet-light`
or `--preprocess starlet` on `evaluate` or `predict`. Both apply the same
two-level transform to sky and queries; neither is the default. On all labelled
scenes, they scored 0.4034 and 0.2967 respectively versus 0.4549 for `none`.

`--joint-alternatives` enables an experimental geometric reassignment over
alternate verified patch locations. It scored 0.4441 on training versus 0.4549
for the default, so it remains opt-in.

`--hough-identification` enables a slower, copy-aware global pose search and
figure-patch assignment. It uses only the supplied skies, queries, and pattern
drawings; it never reads training labels during prediction. On the three
labelled scenes, its mean local score is **0.7901** versus **0.4549** for the
fast default. This is a development result, not a Kaggle score or proof of
hidden-scene performance. The user submitted the generated validation CSV and
reported a **0.456 public leaderboard score**, showing substantial failure to
generalize; do not treat 0.7901 as an expected Kaggle score. It is opt-in
because full-scene inference is several
minutes per scene on this CPU and the confidence gate has only three labelled
scenes for checking.

```powershell
.\.venv\Scripts\python.exe -m constellation evaluate --hough-identification --output outputs/hough-my-evaluation
.\.venv\Scripts\python.exe -m constellation predict --hough-identification --output outputs/hough-my-predictions
```

See [BENCHMARKS.md](BENCHMARKS.md) for per-scene evidence and remaining errors.

An experimental broad-star re-ranking variant is available with
`--hough-identification --bright-star-weight 1`. It kept the same labelled
training score but widened two correct-name margins. Its separate validation
file is `outputs/hough-broadstar-validation/submission.csv`; the public score
reported for that file is **0.53644**, versus **0.456** for the earlier
submission. The mirror, small-figure, and star-seeded
Hough switches are separate experiments, not part of this candidate.

A full-resolution candidate, now publicly scored, is
`outputs/hough-fullres-patchonly-validation/submission.csv`. It keeps the
broad-star identification stage and adds high-confidence full-resolution
masked-NCC patch-location corrections. It was generated end to end with:

```powershell
.\.venv\Scripts\python.exe -m constellation predict --hough-identification --bright-star-weight 1 --fullres-ncc --fullres-cache-dir outputs/fullres-cache-shared --output outputs/hough-fullres-patchonly-validation
```

The run took about 2 h 42 min on this device. Its schema and coordinates
were independently checked. The user submitted this file and reported a
**0.62054 public score**, making it the best scored submission so far. The
previous 0.53644 file is untouched. Full-resolution candidate search
uses supplied sky/query images at inference; synthetic data is used only
for tests. See [BENCHMARKS.md](BENCHMARKS.md) for local evidence and limits.

A newer **unscored** candidate is
`outputs/hough-fullres-multichannel-validation/submission.csv`. It adds a
raw-image check to rerank cached full-resolution locations and changes 16
patch cells relative to the 0.62054 file. Generate it in seconds from the
saved broad-star submission and content-checked candidate cache with:

```powershell
.\.venv\Scripts\python.exe -m constellation.multichannel_candidate --raw-weight 0.5 --output outputs/hough-fullres-multichannel-validation
```

The three-scene local replay rose from 0.85510 to 0.86607; this is not a
guarantee of a better Kaggle score.

## User-provided high-scoring reference

The extracted `demo.zip` is preserved under `demo_reference/code_submission/`,
including its README, source, and selected CSV. That bundle's
`demo_reference/code_submission/submission_selected.csv` received a
**0.95248 public score**. This is a separate reference implementation, not a
score achieved by the original `constellation` package. Its CSV has been
checked against the supplied validation template for row IDs, columns, patch
counts, tuple syntax, coordinate bounds, and constellation names.
An identical convenience copy for submission is at
`outputs/demo-v8-selected/submission.csv`; both files have the same SHA-256.

The reference method combines broad rotated/scaled patch search, multichannel
registration, duplicate/copy-move checks, and geometric fitting of the 48
supplied constellation drawings. It uses no pretrained weights or outside star
catalogue. To reproduce it locally without modifying the selected CSV:

```powershell
.\.venv\Scripts\python.exe demo_reference\code_submission\constellation_v8.py --data data\constellation-detection\participant --split train --eval --out outputs\demo-v8-train.csv --cache outputs\demo-v8-cache --no-fb-extra
.\.venv\Scripts\python.exe demo_reference\code_submission\constellation_v8.py --data data\constellation-detection\participant --split validation --out outputs\demo-v8-validation.csv --cache outputs\demo-v8-cache --no-fb-extra
```

The first command is an independent local reproduction check on the three
labelled scenes. The second regenerates validation predictions from images and
may take substantial CPU time. Keep the provided selected CSV and any newly
generated CSV separate: differences can result from dependency versions.
On this device, the first labelled scene (`pisces`) reproduced a score of
**0.957** in about 32.5 minutes. The optional replay was stopped after that
scene; its `evid/` and `aug/` cache entries remain, but no complete train or
validation CSV has been regenerated yet. The public 0.95248 result refers to
the user-provided selected CSV, not this partial replay.

## Robust method in the package

The reference method has also been ported, with unchanged matching/decision
numerics, to `constellation/robust_engine.py`. The package
runner validates every result against the competition template and image
bounds before naming a complete output `submission.csv`. It never reads the
selected reference CSV during inference. It writes `robust-checkpoint.csv`
after each scene; this partial file must not be submitted. Run it with:

```powershell
# Evaluate all three labelled scenes. The earlier cached pisces scene can be reused.
.\.venv\Scripts\python.exe -m constellation robust-evaluate --robust-cache-dir outputs\demo-v8-cache --output outputs\robust-train

# Smoke-test one validation scene without making a submission CSV.
.\.venv\Scripts\python.exe -m constellation robust-probe --scene constellation_01 --robust-cache-dir outputs\robust-cache --output outputs\robust-probe-01

# Generate and validate a new full submission from sky/patch pixels.
.\.venv\Scripts\python.exe -m constellation robust-predict --robust-cache-dir outputs\robust-cache --output outputs\robust-validation
```

The current full run logs progress in `outputs/robust-validation/run.stdout.log`.
Read its latest lines with `Get-Content outputs\robust-validation\run.stdout.log -Tail 10`.
`robust-checkpoint.csv` contains only completed scenes; `submission.csv` is
created after all 16 scenes pass validation.

The integrated CLI reproduced `pisces` at **0.9570** using the locally
generated evidence cache. A fresh `constellation_01` validation probe named
`ophiuchus` and matched all 23 patch cells of the selected reference row
exactly. The full 16-scene validation run then completed from image pixels,
producing `outputs/robust-validation/submission.csv` (668 patches). It passes
the competition schema, name, tuple, and image-bound checks. Compared with
the selected reference CSV, all 16 constellation names and 666 of 668 patch
cells match exactly. The two differences are a 2-pixel vertical position for
`constellation_10` `patch_12` and one extra present prediction for
`constellation_11` `patch_34`. This new file has **not** been submitted or
assigned a public leaderboard score; the 0.95248 score belongs only to the
user-provided selected CSV.

### Implementable channel-weight experiment

The integrated robust engine now accepts `--robust-channel-weights RAW BANDPASS HIGHPASS`
for its final exact-match candidate ranking. The default `1 1 1` preserves the
reference behavior. This is an experimental setting, not a new trained model
or a change to the saved baseline submission. For example:

```powershell
.\.venv\Scripts\python.exe -m constellation robust-evaluate --scene pisces --robust-cache-dir outputs\demo-v8-cache --robust-channel-weights 1.5 1 0.5 --output outputs\ablation-raw-pisces
```

On all three labelled scenes, the equal-weight baseline scores 0.96162 on
average. Raw-heavy `1.5 1 0.5` scores 0.96398: `pisces` improves from 0.95703
to 0.96448 and `scorpius` from 0.95747 to 0.96461, but `taurus` regresses from
0.97037 to 0.96287 because of an extra false positive. The milder
`1.25 1 0.75` scores 0.96183 overall and regresses on `scorpius`. The
high-pass-heavy `0.5 1 1.5` scores 0.95703 on `pisces` (no gain). These small,
mixed results from only three training scenes do **not** justify changing the
default.

The raw-heavy validation candidate is
`outputs/robust-rawheavy-validation/submission.csv`. It passes the submission
validator, retains all 16 constellation names, and changes 35 patch cells
relative to the equal-weight output, including 27 present/absent decisions.
The user submitted this exact file and reported a **0.92 public score**,
below the user-provided reference's **0.95248**. Do **not** submit the
raw-heavy candidate again. The currently recommended, already-scored file is
`outputs/demo-v8-selected/submission.csv`, an SHA-256-identical copy of
`demo_reference/code_submission/submission_selected.csv`. Local train gains
are not a reliable model-selection signal for this three-scene dataset. Each
run records its exact weights and scores in `robust-report.json`.

An exploratory audit considered rejecting figure-star assignments whose
selected match was far below the patch's best match. Three wrong training
assignments had that pattern, but so did 13 unlabeled validation assignments.
The annular check did not cleanly separate dim valid stars from false matches,
so that broad rule was not added to inference.

### Final conservative fusion after the 0.92 result

`constellation/conservative_fusion.py` combines two locally generated robust
runs using only sky/query pixels and their cached match evidence. It starts
from the equal-weight output. It removes a weak, fourth-or-worse figure-star
claim when independent high-pass ring detail is poor; it can transfer a star
to a raw-heavy-only patch when that patch's match and ring detail are strong
and it wins against the incumbent at the same location. It does **not** read
the selected reference CSV or training labels during validation inference.

The full three-scene labelled replay scores **0.96667** versus **0.96162** for
equal weights and **0.96398** for raw-heavy, with no scene-level regression.
The validated 16-scene file is `outputs/final/submission.csv`.
It changes only three patch cells relative to the equal-weight run: a patch
ownership transfer in `constellation_11`, and one weak-match abstention in
`constellation_12`. In a post-inference parity check it differs in five patch
cells from the already-scored 0.95248 reference file. The user submitted this
fused file and reported a **0.94 public score**. Use the notebook at the top of
this README to reproduce the final file end to end.

The `--output` directory for `audit` receives `patch-audit.json`: hardware and template counts, per-patch brightness/blur/information flags, exact file and pixel duplicates, and similar-looking pairs that still keep separate IDs. Extra contrast or denoising is measured on a few labelled patches and left disabled.

The basic example writes `outputs/my-predictions/submission.csv`; the newer
candidate path is listed above. Nothing is uploaded to Kaggle. The `--output`
directory for evaluate/predict receives checkpoints after each scene,
per-patch confidence diagnostics, a JSON report, and predicted coordinates.
A final submission filename is created only after all scenes complete and
its row IDs, patch counts, class names, and coordinates pass validation.
Reuse of an output directory overwrites its matching result files; choose
a new directory for comparisons.

All paths are relative to the project except `--data`, whose default is resolved relative to the source code. To use a separately supplied dataset:

```powershell
.\.venv\Scripts\python.exe -m constellation predict --data C:\path\to\participant --output outputs/new-scenes
```

Prediction requires `patterns/`, `validation/`, and `sample_submission.csv` in that root. It does **not** load or require training images or labels. Scene folder names and patch counts are used only to map files into the required output schema.

## Build in a fresh Python environment

Use Python 3.11 or newer; the installed configuration was tested on Python 3.12.14 / Windows x64.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records the package versions used for the local benchmark. It can replace `requirements.txt` when those exact wheels are available. No credentials are needed once the dataset has been downloaded.

## What the implementation does

1. Detects up to 35,000 source candidates using a difference of Gaussians (Gaussian blur as a linear filter) plus Harris corners, with subpixel peak interpolation. The optional regular grid covers patch centres that are not near strong peaks.
2. **SIFT + RANSAC is the alignment.** Distinctive RootSIFT matches vote for a similarity (translation, rotation, uniform scale). That mapped patch centre is the location. Polar-descriptor retrieval runs only when RANSAC has no consensus.
3. Photometric verification refines that pose (or the polar fallback) against image pixels. The patch is the template filter. Radial-profile subtraction reduces the influence of an uninformative bright central star.
4. Extracts white star disks from each RGBA reference template. Four-point triangle-area ratios propose **affine** alignments (non-uniform scale, shear, reflection). RANSAC drops outlier correspondences; least squares refits the inliers. A full homography is not used: these are star coordinates, not a perspective scene pair.
5. Emits `(x, y, m)` or `-1` for each patch, and the highest-ranked constellation name (or `unknown` when a hypothesis cannot be built).

Patch files are used as supplied. The matcher subtracts a radial background and normalizes contrast; additional CLAHE, gamma, or denoising is not applied. `audit` can distinguish exact duplicates from similar-looking patches, but every patch ID still receives its own prediction.

The default appearance threshold is 0.80, with the margin gate disabled. These are **development choices** informed by training diagnostics, not independently validated universal thresholds. `--threshold` and `--margin` expose them for experiments. The current score is a correlation, not a calibrated probability. Scores across different patches need not be directly comparable.

## Diagnostics and ablations

```powershell
# Small smoke test. Its score is explicitly labelled partial.
.\.venv\Scripts\python.exe -m constellation evaluate --scene taurus --limit-patches 8 --skip-identification --output outputs/smoke

# Separate point-set matching from localization errors (training labels only).
.\.venv\Scripts\python.exe -m constellation oracle-identify --output outputs/oracle

# Test detector coverage and verification given oracle training locations.
.\.venv\Scripts\python.exe -m constellation.diagnose

# Select thresholds on two scenes and inspect the held-out third scene.
.\.venv\Scripts\python.exe -m constellation.calibrate outputs/my-evaluation/evaluate-patches.json --output outputs/calibration.json
```

Threshold-only held-out results are **not** a clean held-out model evaluation: the representation was developed while examining all three training scenes. The calibration ablation holds constellation identification at `unknown`; its maximum weighted score is 0.70. Oracle diagnostics are separate modules/commands and are never used during inference.

`--no-sift` disables the additional feature proposal source. `--sources`, `--top-k`, and `--grid-step` control retrieval cost. `--threads` caps OpenCV threads; the CLI also defaults numerical libraries to eight threads unless their environment variables already specify a value. CUDA is not currently used even if an NVIDIA GPU is present.

## Metric and limits

The local evaluator follows the supplied textual formula: 25% presence macro-F1, 20% localization, 25% greedy one-to-one figure recovery, and 30% name accuracy, averaged equally over scenes. Localization reward is 1 at distances up to 12 px, linear to 0 at 36 px. Geometry considers every predicted present point, independent of its patch ID or membership bit. No official Kaggle scorer source was provided; zero-denominator F1 and empty-target conventions are documented in `constellation/metrics.py` and may differ from the server.

The first implementation is a measurable baseline, not a finished competitive solution. Remaining issues:

- Repetitive or very degraded patches can match the wrong location; a high correlation is not sufficient evidence on its own.
- The source detector misses many true centres, and the regular grid trades much more computation for better coverage.
- Affine quad ranking can prefer a chance alignment in clutter. An oracle probe on Taurus selects Orion when all true present points are included, even though it selects Taurus with just true figure stars.
- References with fewer than four extracted nodes cannot generate an affine quad hypothesis. Under an unconstrained affine model, tiny partial figures are intrinsically ambiguous; additional image evidence or a stronger transformation prior is needed. All 48 templates are loaded; the current naming stage does not have equal coverage across them.
- Only the 40 most confident present locations are used in the global pattern search to bound quartic hypothesis generation. This can discard weak but relevant figure stars in crowded scenes.
- A few local development scenes do not establish unseen-scene performance. Public leaderboard feedback should not substitute for controlled validation.

See `PLAN.md` for milestones, `BENCHMARKS.md` for measured results, and the JSON diagnostics for individual errors.
