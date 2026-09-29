# First local baseline: measured results

Measured on this device using the copied competition dataset: Python 3.12.14, 32 logical CPU cores, and an available NVIDIA RTX 5070 Ti Laptop GPU (12,227 MiB). This implementation uses the CPU. Exact package versions are recorded in `requirements-lock.txt`.

These are **training/development results**, using the evaluator reconstructed from the written brief. They are not a Kaggle leaderboard score or an unbiased estimate of unseen-scene performance.

## Selected default: fast

Command: `.\.venv\Scripts\python.exe -m constellation evaluate --output outputs/baseline-fast`

- Mean weighted score: **0.454882** across three equally weighted training scenes.
- Presence macro-F1: 0.723613. Localization: 0.425451. Geometric recovery: 0.355556. Identification: 1/3.
- Per-scene score: Pisces 0.787439; Scorpius 0.323767; Taurus 0.253441.
- All 116 training patches processed in **26.5 seconds**.
- Parameters: 35,000 detected positions, 60 candidates per scale, six scales, eight refined locations per patch, RootSIFT proposals enabled, correlation threshold 0.80, margin threshold 0.

Saved evidence: `outputs/baseline-fast/evaluate-report.json`, `evaluate-patches.json`, and `evaluate-predictions.csv`.

For comparison, predicting every training patch absent and every name unknown scores **0.070200** under this local evaluator. This is the sample-style floor, **not** the official competition baseline to beat.

## Optional dense search

Command: `.\.venv\Scripts\python.exe -m constellation evaluate --grid-step 4 --top-k 100 --output outputs/baseline-thorough`

- Mean weighted score: **0.407206**.
- Presence macro-F1: 0.726831. Localization: 0.544160. Geometric recovery: 0.466667. Identification: 0/3.
- Per-scene score: Pisces 0.423305; Scorpius 0.396844; Taurus 0.401471.
- All 116 patches processed in **235.9 seconds**, partly concurrent with validation inference.
- Searches 585,564 positions per 3000 × 3000 scene; descriptor extraction is streamed in batches.

Dense retrieval improves average localization and geometry but loses the successful Pisces identification, reducing the total score. It therefore remains an experiment rather than the default. More candidates alone do not resolve ambiguous patch matches or cluttered global alignments.

During the simultaneous benchmark/inference runs, the active Python workers each showed roughly **2.1 GiB peak working set**, including OpenCV's SIFT computations. This was a process observation, not a controlled isolated memory benchmark.

## Validation output

Command: `.\.venv\Scripts\python.exe -m constellation predict --output outputs/draft-fast`

The fast profile produced `outputs/draft-fast/submission.csv` for **16 scenes / 668 patches / 90 columns** in **180.8 seconds**, while the dense training benchmark was also running. Independently checked:

- Exact required scene ID set, with each scene appearing once.
- Exact sample CSV column order and unchanged patch counts.
- Valid pattern names or `unknown`, finite coordinates inside the associated images, and binary membership values.

Validation labels are unavailable, so no accuracy or score is claimed for this draft. It has not been submitted to Kaggle.

## Scientific diagnostics

The affine pattern matcher ranks the correct name first on all three scenes when supplied only true figure points. With every true present point included (figure + off-figure), it ranks Pisces and Scorpius first but ranks Taurus sixth, behind Orion. Thus even perfect localization does not solve the remaining naming ambiguity.

Among 71 truly present training patches, the 35,000-position source detector places a candidate within 3 px of the labelled centre for 18/27 Pisces patches, 10/26 Scorpius patches, and 10/18 Taurus patches. Searching only strong peaks misses many useful centres. An oracle-location test nevertheless produces high median verified correlations (0.888, 0.896, and 0.857 respectively), showing that retrieval coverage is an actionable weakness. These oracle values are diagnostics, never inference inputs.

Threshold sweeps select settings on two scenes and report the third, but the representation itself has been developed on all three scenes. They are explicitly labelled **threshold-only** held-out analyses. Configuration 0.80 / 0.0 was chosen from development results.

## Verification

**11 tests pass**, including the 12–36 px reward ramp, absent-class handling, geometry's one-to-one rule, coordinate order, affine descriptor invariance including reflection, recovery of a synthetic rotated/scaled patch, blank-patch rejection, submission validation, and an actual prediction command on an unseen synthetic dataset without any training labels.

## Next experiments

1. Preserve separate retrieval pools for detected stars and grid positions, and apply spatial nonmaximum suppression before the refinement budget. Nearby grid proposals currently can displace more informative alternatives.
2. Retain several verified locations per patch, then let constellation hypotheses resolve ambiguous local matches while preserving independent photometric evidence for presence.
3. Improve global model selection using withheld correspondences, transform plausibility, uncertainty, and background chance-match likelihood. Taurus's oracle clutter failure is the diagnostic to address.
4. Add coverage for small reference figures and explicitly test the assumed transformation family. Fewer than four nodes cannot support the present affine quad descriptor.
5. Profile retrieval separately before adding GPU execution. A learned GPU descriptor should justify itself through improved retrieval recall and scene-held-out tests.

## Cloud removal and ambiguity follow-up (2026-09-28)

See [CLEANING.md](CLEANING.md) for full measurements and saved reports.

- Difference-of-Gaussians broad background sigma: 4 => 0.3044, 7 => 0.4549, 10 => 0.3728, 14 => 0.3597, 20 => 0.4016, 28 => 0.3137. Sigma 7 remains the default.
- In-frame node / patch-count prior: no name accuracy gain for tested widths and weights.
- Relative-best-match-gap presence gate: best calibration used gap zero.
- CUDA: system GPU present; OpenCV CUDA device count 0, PyTorch and CuPy absent.
- Retaining multiple verified locations exposed a correct <=12px alternative for 4 additional true patches. Geometric reassignment from the top-1 pose scored 0.4441 vs 0.4549 and did not change names; it remains experimental.

These experiments remain below the requested 0.9. The next structural opportunity is to generate global pattern poses from the alternative locations themselves, then test with broader labelled evaluation.

## Copy-aware global pose experiment (2026-09-28)

The structural opportunity above is now implemented as opt-in
`--hough-identification`. It searches rotated/scaled versions of each query at
3/8 sky resolution, retains multiple direct-match positions, screens
similarity poses of all supplied constellation drawings against a spatial
evidence raster, and refines promising poses with unique query-to-node
assignment. A whole-sky annular search supplies displacement-consistent
copy-neighborhood evidence. The pose score penalizes unmatched in-frame nodes
and applies a broad prior to the ratio of query count to in-frame nodes. A
margin gate falls back to the original name when the global winner is weak.
Accepted poses can promote and relocate issued queries, but no labelled
coordinates or scene names are used by the inference path.

Actual CLI evaluations, with all issued patches in each scene:

- Pisces: **0.787439** (same as original); global alternative had a 0.79
  top-two name margin, below the 1.2 acceptance threshold.
- Scorpius: **0.760258**, up from **0.323767**; identification correct,
  geometry 0.70, presence 0.772, localization 0.462.
- Taurus: **0.822549**, up from **0.253441**; identification correct,
  geometry 1.00, presence 0.824, localization 0.333.
- Equal-scene mean: **0.790082**, up from **0.454882**. These are scores from
  our reconstructed local metric, not the official leaderboard.

Evidence: `outputs/hough-final-train/evaluate-report.json`, produced by an
end-to-end run of the final 200-hypothesis-cap configuration over all three
training scenes in **628.2 s**. The mean was **0.790082** with per-scene
scores matching the values above. Earlier exploratory runs used all coarse
hypotheses or a 500-hypothesis cap; the final run supersedes those checks.

The full 200-cap validation prediction completed in **3,646.8 s** (CPU).
Its `outputs/hough-validation/submission.csv` passed an independent contract
check: **16 scenes / 668 patches / 90 columns**, the exact sample header,
allowed constellation names and in-bounds coordinates. Against the original
fast draft it changed two scene names and 70 patch cells, including 19
absent-to-present promotions. Validation has no local accuracy labels;
no accuracy score is claimed and the CSV has not been uploaded to Kaggle.

Diagnostic controls that did **not** justify deployment:

- Searching 500 instead of 60 polar candidates per scale did not improve the
  three-scene mean; expanding detected source positions to 100,000 only moved
  it to roughly 0.463 at a substantial runtime cost.
- Foreground Gaussian scale 1.5, the phase-aware polar descriptor, and
  per-scene diagonal descriptor whitening did not materially improve the
  complete score. They remain optional ablations, not selected defaults.
- Direct top-1 pixel correlation was correct on only 2/18 present Taurus
  patches, although the top-30 candidate set covered 15/18. One-to-one site
  assignment and hub penalties alone did not fix the ranking. A simple
  locally trained pair reranker gained a few correct locations but did not
  improve the full scene-held-out score; it was not deployed.
- Exact 16x16 copy-window search found 2,108 verified pairs in Pisces but none
  in Scorpius or Taurus. Approximate annular matching was needed for the
  latter scenes. Under NCC >= 0.90 and displacement-bin support >= 5, close
  anchors occurred at 1/10, 1/10, and 3/6 issued figure stars respectively,
  with zero off-figure anchors within 10 px in these three training scenes.
  This tiny sample does not establish a zero false-positive rate on validation.
- Bright-star support on saved Hough poses did not transfer as hoped: adding
  one point per Gaussian-weighted match to the strongest 300 detected stars
  (20 px tolerance) demoted the correct Scorpius hypothesis from rank 1 to 3
  and made an Orion decoy the winner in Pisces. At 3,000 stars / 40 px,
  chance support for large figures weakened margins. This remains a diagnostic
  probe, not an inference feature.

The method is still below the requested 0.9. Localization is the largest
remaining weakness, especially for dim or heavily softened off-figure
queries that global figure assignment cannot rescue. Only three labelled
scenes exist here, so the apparent gain and name-margin gate are vulnerable
to development-set selection; the private-set score is unknown.

## Public-score feedback and follow-up (2026-09-28)

The user confirmed uploading `outputs/hough-validation/submission.csv` and
reported a **0.456 public leaderboard score**. This supersedes any suggestion
that the 0.790082 training score predicts unseen performance. Kaggle does not
provide scene or patch labels here, so the score cannot be decomposed into
presence, localization, geometry, and naming errors locally.

Evidence of a distribution shift: the median saved match score is **0.743**
on the 116 training patches and **0.641** on the 668 validation patches;
predicted-present rates are **0.509** and **0.386** respectively. This suggests
under-recall but cannot establish the validation truth prevalence. Lowering
the presence threshold from 0.80 to 0.70 reduced training mean score from
0.790082 to **0.777808**; forcing 60% of patches present yielded **0.773105**.
Do not ship either change merely to respond to the public score.

The user-supplied method diagram motivated two test-only synthetic regressions:
a reflected schematic constellation, and a rotated/scaled 32x32 crop with
blur, sensor noise and smooth brightness drift. Synthetic examples are created
only in tests; the inference pipeline reads the supplied images and patterns.
Mirror-aware Hough search passed the synthetic test but reproduced the same
three training scores as the original method. A wider 32-candidate Powell
refinement reproduced the same Taurus pre-geometry score of **0.253441**.
The combined log-blur/residual score raised correct top-choice ranking from
33/37 to 35/37 among training patches whose true location was already in the
saved localizer alternatives; candidate coverage, not just re-ranking, is the
larger weakness. Fully verifying 30 direct-search candidates on Taurus found
only 3 correctly localized present patches, versus 6 with the original
localizer, so that route is not deployed as-is.

Combining dense 4-pixel grid retrieval with the Hough stage improved Taurus
localization from 0.333333 to **0.555556**, but reduced presence from
0.823529 to 0.792502 and geometry from 1.0 to 0.833333. Its total Taurus
score was **0.817570**, slightly below the 0.822549 fast-Hough result. The
other two scenes scored **0.755712** (Pisces) and **0.767950** (Scorpius),
for an equal-scene mean of **0.780411** versus **0.790082** without the dense
grid. Dense retrieval is not selected.

The six-hit global Hough minimum excludes 21 of 48 drawings with fewer than
six nodes. An opt-in `--small-hough` run adds four- and five-node hypotheses;
it reproduced the same **0.790082** three-scene training mean without a
measured gain. Two- and three-node figures remain intrinsically ambiguous
under point-only geometry and need additional evidence. No new validation
CSV has been generated from these unproven experiments.

A broader DoG star detector (`sigma=4` minus `sigma=12`) finds a top-300
peak within 20 px of **10/10**, **10/10**, and **5/6** labelled figure stars
in Pisces, Scorpius, and Taurus respectively. Uniform random positions hit
the same catalogue about 3–4% of the time in a 1,000-point probe. The
previous sharp DoG detector missed nearly all large saturated figure stars.
Adding broad-star support at weight 1 to global pose scoring kept the
three-scene mean at **0.790082**, but widened correct Hough name margins from
1.45 to **6.75** (Scorpius) and 2.58 to **6.77** (Taurus). In Pisces its
incorrect global winner still had only a 0.56 margin and was rejected.
This is opt-in through `--bright-star-weight 1`; a separate validation CSV
was generated at `outputs/hough-broadstar-validation/submission.csv` in
**3,526.9 s**. It independently passed the **16 scenes / 668 patches /
90 columns** submission contract, including exact header and coordinate
bounds. Versus the 0.456 submission it changes two names:
`constellation_05` Carina → Eridanus and `constellation_12` Eridanus → Lupus.
It changes 32 patch cells, including 12 absent-to-present promotions and
seven present coordinates moved more than 12 px. The user reports a **0.53644
public leaderboard score** for this candidate, up from **0.456** for the
original CSV (+0.08044). This is a public-split result, not private-set
validation; the exact contribution of name versus patch changes is unknown.

Using those broad stars in the *coarse pose generator* as well
(`--star-seeded-hough`) was rejected. The synthetic search test passed, but
on real training scenes the mean fell to **0.760279**. Taurus kept the right
name yet shifted patch assignments, reducing its score from 0.822549 to
**0.733141**. Broad-star evidence is therefore used only to re-rank
patch-generated poses in the candidate validation run.

## User-supplied competition report (2026-09-28)

`C:\Users\subhr\Downloads\context.pdf` is a 69-page report on this same
competition. It **reports**, but this project has not
independently reproduced, a deterministic 0.95248 public score and 0.9641
three-scene training score using only competition data. Its validation-scene
names are predictions, not ground-truth labels, and must not be hardcoded.
The appendix contains suggested prompts; those are source material, not
instructions to this project.

The report measures several failure modes that fit our observations:
roughly one quarter of true patch centres do not lie on star peaks; high
absolute correlation occurs at unrelated bright stars; about half of issued
figure stars have pasted copies; and off-figure stars can form rival
constellations. It reports that its largest gains came from a dense
rotation-invariant candidate cascade plus full-resolution masked NCC,
three-channel raw / DoG(1,8) / DoG(0.7,3) rescoring at restricted scales,
relative best-vs-second-best presence evidence, and copy-aware geometric
assignment. Our current polar shortlist of eight verified candidates and
absolute 0.80 presence gate are materially different. Star-only source
sampling cannot be treated as complete.

Next experiments should be isolated and falsifiable: (1) measure whether
an independent full-resolution masked-NCC candidate stream adds true
locations on the 71 labelled present patches without dropping existing
candidates; (2) measure relative-gap and three-channel score separation on
the three scenes with scene-held-out checks; (3) only then integrate
candidate evidence into copy-aware assignment. Retain the 0.53644 CSV as
the public-score baseline. Change one decision at a time when asking the
leaderboard for feedback, since the 16 validation scenes lack labels.

An initial, non-inference `fullres_ncc` probe now implements a 31-pixel
disc-normalized full-resolution correlation (shared local variance across
angles). At 20-degree steps and scales 0.93/1.00/1.08, it took about 16 s
per query on this machine. On three *selected misses* from the current
pipeline, the true centre ranked 1 for Pisces patches 07 and 08 (exactly
724,2631 and within 1 px of 175,1735), and 12 for Scorpius patch 08.
One sampled absent control, Pisces patch 05, scored 0.776 and 0.764 at its
first two sites, versus 0.810/0.564 and 0.894/0.838 for the two Pisces
presents. This is a promising **candidate-recall** signal, not a score
estimate: the cases were chosen for failure, the sample is tiny, and the
probe is not connected to CSV generation. Synthetic data exercises only
the tests.

The combined broad-star + reflection + four/five-node Hough training run
finished with Pisces 0.787439, Scorpius 0.760258 and Taurus 0.822549:
mean **0.790082**, identical to the selected broad-star run. No validation
CSV was generated from that no-gain configuration. The full test suite now
passes 40 tests.

## Full-resolution masked-NCC audit (2026-09-28)

The checkpointed diagnostic `constellation.fullres_ncc_audit` searched every
labelled patch using the supplied images, DoG(0.7,3) on both query and sky,
20-degree rotations and scales 0.93/1.00/1.08. It took **1,351.9 s** for all
116 patches, about 11.7 s each. The correct location was rank 1 for
**46/71** present patches and in the top 30 for **66/71**; the existing
localizer's saved alternatives contained **37/71**. By scene, top-30
coverage was 26/27 Pisces, 25/26 Scorpius and 15/18 Taurus. It included
all 10 Pisces and 10 Scorpius issued figure stars, and 5/6 Taurus stars.
The Taurus misses show why this narrow-scale stream must be a union with
the existing broader-scale path, not a replacement. The median relative
best-vs-runner-up gap was 0.156 on 71 presents versus 0.021 on 45 absents.
One absent in each of Pisces and Scorpius exceeded a 0.16 gap; Taurus had
none.

Offline replay over the frozen broad-star training CSV changed only
non-Hough-assigned patches with relative gap at least 0.16. The per-scene
scores became **0.885847 Pisces, 0.775642 Scorpius, 0.903819 Taurus**:
equal-scene mean **0.855103**, up from 0.790082. Thresholds selected from
the other two scenes and applied to the held-out scene gave mean **0.841835**
(.10 for Pisces, .16 for Scorpius and Taurus). This is still a three-scene
estimate and cannot establish public/private performance. Full-resolution
search and an optional top-five candidate union for Hough are behind
`--fullres-ncc` and `--fullres-hough`, with the former's default gap 0.16.
The default pipeline and best public CSV remain unchanged until an
end-to-end run is validated.

The first full end-to-end test of **unfiltered top-five NCC candidates in
Hough** scored 0.885847 Pisces, 0.751322 Scorpius and 0.828819 Taurus:
mean **0.821996**. The name stayed correct in all three scenes, but extra
flat candidates shifted the geometric pose. Scorpius's Hough star hits
fell from 9 to 8 and its name margin from 6.75 to 1.96; Taurus geometry
fell from 1.0 to 0.833. This is worse than the post-Hough patch-correction
replay's 0.855103, so `--fullres-hough` is not selected for validation.
An unsupported-node likelihood scorer is now opt-in with
`--likelihood-hough`; it must be tested separately before use.
The targeted Scorpius test raised the Hough name margin to **5.75** but
selected the same inferior pose and scored **0.751322**. It is not selected.
The validation candidate retains the established broad-star Hough stage and
adds only the post-Hough masked-NCC correction at relative gap 0.16. Its
full end-to-end run finished in **9,735.6 s** (about 2 h 42 min) and wrote
`outputs/hough-fullres-patchonly-validation/submission.csv`. An independent
CSV audit found 16 scenes, 668 patches, the exact 90-column sample schema,
valid pattern names, and finite coordinates within the 3000 x 3000 skies.
Compared with the 0.53644 broad-star submission, the new CSV changes 162
patch cells: 64 absent-to-present promotions, no present-to-absent removals,
two coordinate shifts over 12 px, and no constellation-name or membership
changes. All 45 tests pass. These are label-free checks, **not** a public
score; the old submission remains the only publicly scored file, and it was
not overwritten. The new candidate uses supplied competition data only;
synthetic images were limited to tests.

## Public feedback and multichannel follow-up (2026-09-29)

The user submitted `outputs/hough-fullres-patchonly-validation/submission.csv`
and reported a **0.62054 public score**, improving on the earlier 0.53644.
That file is now the best publicly scored submission and remains untouched.

A confidence-gated Hough candidate union (only the top full-resolution site
when its relative gap is at least 0.16) reproduced the three post-Hough
training scores exactly: Pisces 0.885847, Scorpius 0.775642, Taurus
0.903819. It offered no measured gain and was not used for the next CSV.
A direct affine constellation-name ranker applied only to high-gap NCC sites
also failed to rank the correct name first on the labelled scenes, so it was
not used.

An independent raw-image NCC check reranked the cached top-30 DoG sites. Its
top-one correct-location count on the 71 labelled present patches rose from
46 to 51 with raw weight 1.0. In a frozen-submission replay, weights 0.5 or
1.0 with relative gap 0.16 gave Pisces **0.900364**, Scorpius **0.775642**,
and Taurus **0.922213**: mean **0.866073** versus 0.855103 without the raw
channel. This is a small three-scene gain, not a validation score. A separate
validation candidate at weight 0.5 is
`outputs/hough-fullres-multichannel-validation/submission.csv`. Its weight-0
reconstruction was byte-identical to the 0.62054 submission. The weight-0.5
candidate differs from that scored file in only 16 patch cells (eight
present-to-absent removals, one promotion, and seven location adjustments);
names and membership flags are unchanged. The CSV contract and coordinate
bounds pass; all 48 tests pass. Its public score is unknown until submitted.

The label-free confidence-gated Hough audit was also run on all eight
validation scenes whose original Hough result fell below its acceptance
margin. Seven kept exactly the same name and ranking. Only
`constellation_03` changed from the baseline Ursa Major to a Perseus
hypothesis (margin 2.32, six assigned patches). The three labelled training
scenes provide no reliable calibration for that intermediate margin:
Pisces' wrong top Hough name had margin 0.09, while the correct Scorpius and
Taurus names had margins 6.72 and 7.82. This one uncertain name override is
**not** included in the multichannel submission candidate.
