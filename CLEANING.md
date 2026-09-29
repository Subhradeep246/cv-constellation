# Non-destructive cleaning and end-to-end experiment

Run date: 2026-09-28. Original images and competition query IDs are preserved.
This pass implements quality auditing, derived normalized views, exact-query
computation reuse, and controlled preprocessing experiments. It does **not**
implement the entire external playbook or claim its reported leaderboard score.

## What was implemented

- `python -m constellation.cleaning --output outputs/cleaning-e2e` checks all
  patch dimensions/decodability, measures brightness and dynamic range, exports
  background-subtracted RMS-normalized float32 patches, and hashes every source
  file before and after the operation. Output must be outside the original data.
- `normalized/<split>/<scene>.npz` contains `patch_ids` and `normalized` arrays.
  These are diagnostic/experimental views, **not** an alternative competition
  data root. The production matcher continues to load the originals and apply
  its existing difference-of-Gaussians/contrast normalization internally.
- Exact duplicates are identified by decoded pixels. Inference reuses matching
  computation only for identical prepared queries **within the current scene**,
  then expands the results back to every issued query ID. No cross-scene
  coordinate sharing or label propagation occurs.
- Copied-neighborhood candidates are searched independently of query patches.
  The top 300 DoG peaks are compared over annuli of radius 20–48 pixels, avoiding
  the central star. Candidates need raw and high-pass NCC >= 0.85 after a +/-2px
  translation refinement. This is a limited diagnostic, not exhaustive copy-move
  detection: missed dim centers, rotation, scale changes and larger shifts are
  not covered. Candidates are not used to assign figure membership in inference.
- `--preprocess none|gamma|bilateral|clahe|starlet-light|starlet` exposes reproducible ablations. The
  same recipe is applied to the sky and query. CLAHE uses approximately 32px
  tiles on both, not the same tile count at radically different image sizes.
  The starlet modes use a two-level, undecimated B3-spline transform with a
  per-scale MAD noise estimate. They hard-threshold detail coefficients at
  1.5 or 2.5 estimated noise standard deviations, respectively, and retain
  the smooth component. They use no outside training data or learned weights.

## Measurements

All 853 source files passed the integrity comparison. The 784 patches across
19 scenes decoded with the expected 32x32 shape. There were no exact duplicate
patch groups. The separate aligned-patch audit found 255 pairs with NCC >= 0.90,
including 24 same-scene pairs; these are similar appearances, not duplicate proof.

Quality review hints: 46 patches with mean intensity <15; 126 with at least 1%
of pixels at 255; two with p99-p1 <15. Flags overlap. An endpoint count does not
prove sensor saturation; low brightness is not absence; a noise or sharpness
proxy cannot establish degradation independently of scene content. No queries
were removed, relabeled, recentered, or assigned absence from these statistics.

The bounded sky-neighborhood search found 22 candidate pairs. Only one was in
the three training skies (Pisces), so this search does **not** substantiate the
playbook's much higher reported coverage. It requires broader search before
being used as a reliable figure-star signal.

Full development evaluation used all 116 labelled patches, the same matching
parameters and threshold 0.80, and all four score components:

- Existing preprocessing (`none`): mean score **0.454882**; localization 0.425451;
  geometry 0.355556; one of three constellation names correct.
- Bilateral (`5,25,25`): **0.323413**; localization 0.382241; geometry 0.288889;
  no names correct.
- Gamma (`0.6`): **0.271510**; localization 0.275878; geometry 0.222222;
  no names correct.
- CLAHE (`clipLimit=2`): **0.230602**; localization 0.250712; geometry 0.122222;
  no names correct.

Decision: keep the existing preprocessing and save the alternatives as explicit
experimental switches. This is a fixed-configuration ablation, **not** proof
that every parameterization of each technique is inferior. Thresholds were not
retuned for each variant. Three development scenes provide weak generalization
evidence; validation images have no local labels. Some jobs ran concurrently,
so timings in their reports are not clean speed comparisons.

### Starlet preprocessing follow-up

On the same 116 labelled patches across all three training scenes, with the
same matcher settings and decision threshold, `starlet-light` scored **0.403390**
and `starlet` scored **0.296739**, versus **0.454882** for `none`. The light mode
raised Pisces to 0.740897, but lowered Scorpius to 0.244857 and Taurus to
0.224415. Stronger denoising lowered Pisces to 0.339792; its other scene scores
were 0.254812 and 0.295614. The per-scene behavior argues against globally
enabling either mode. In particular, noise-like fine detail may also carry the
relative star pattern needed for localization. These results do not rule out a
separate, selectively fused denoised channel or a scene-adaptive noise model.

Reproduce with `python -m constellation evaluate --preprocess starlet-light
--output outputs/preprocess-starlet-light` or use `--preprocess starlet` and
`outputs/preprocess-starlet`. Reports are saved in those output directories;
source images and query IDs remain unchanged. A synthetic point-source/noise
test plus the full 27-test suite passed.

The [starlet/a trous literature](https://www.aanda.org/articles/aa/full_html/2023/02/aa45345-22/aa45345-22.html)
motivates scale-dependent significance thresholds in astronomical denoising;
it does not imply that thresholding improves this competition's downstream
matching score. This implementation is a measured experiment, not a learned
restoration model.

## Research and interpretation

1. Background subtraction plus normalized correlation is the conservative first
   choice for gain/offset variation. [Lewis, Fast Normalized Cross-Correlation](https://www.scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf)
   develops efficient normalized matching. Our current matcher already includes
   background filtering and normalized correlation, so adding brightening is
   not automatically additional invariance.
2. Astronomical detection should distinguish background from point-source signal.
   [Photutils background estimation](https://photutils.readthedocs.io/en/stable/user_guide/background.html)
   and [point-source detection](https://photutils.readthedocs.io/en/stable/user_guide/detection.html)
   document sigma-clipped background estimation and noise-relative thresholds.
   These are candidate detector experiments, not permission to discard dim queries.
3. [OpenCV's CLAHE documentation](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
   explains local contrast enhancement and noise amplification. Here CLAHE lost
   score in a full run, so a more visually contrasted patch is not necessarily
   a better matching template.
4. [Buades, Coll and Morel, Non-Local Means](https://www.ipol.im/pub/art/2011/bcm_nlm/)
   motivates similarity-based denoising. It does not establish that denoising
   preserves the faint context needed by these 32x32 queries. NLM itself was not
   evaluated here; the bilateral result only applies to that tested filter.
5. [Arandjelovic and Zisserman, RootSIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
   supports descriptor normalization for retrieval. RootSIFT is already part
   of the current pipeline, not a newly demonstrated gain in this experiment.

The playbook's copy evidence and multi-candidate geometric assignment are worth
testing next. Preserve alternate locations rather than deleting duplicates or
forcing an early top-1 answer. The current experiment did not establish the
playbook's no-false-positive claim, transform priors, or reported 0.9513 score.

## Continued experiments

The matcher exposes `--background-sigma` to vary the broad Gaussian subtracted
from each query and sky. The same six scales, threshold, and other settings were
used across all three training scenes:

| Background sigma | Mean local score |
|---:|---:|
| 4 | 0.3044 |
| 7 (baseline) | 0.4549 |
| 10 | 0.3728 |
| 14 | 0.3597 |
| 20 | 0.4016 |
| 28 | 0.3137 |

No broader background removal setting is enabled. Sigma 20 correctly named
Pisces but missed the other two names. The baseline scored best overall.

A bounded patches-per-hypothesized-in-frame-node prior (target ratio 2.97,
widths 0.35–1.05, several weights) did not improve name accuracy from 1/3.
Wrong hypotheses often had similar ratios: Scorpius's top wrong Eridanus fit
yielded 2.56 patches/node, while Taurus's top wrong Sagittarius fit yielded
2.43. A relative uniqueness gate `(best - second)/(1 - best)` was swept from
0 to 0.50 with eight score thresholds. The best baseline setting used no gap.
Ambiguous duplicate locations can belong to true figure stars.

Experiments are retained in `outputs/cloud-sigma-*/evaluate-report.json`,
`outputs/relative-gap-baseline.json` and `outputs/patch-budget-prior.json`.
Default sigma remains 7; use `--background-sigma 20` for the tested cloud
subtraction alternative.

Windows sees the NVIDIA GPU, but this OpenCV build reports zero CUDA devices;
PyTorch and CuPy are absent. The implementation runs on CPU.

Multiple distinct verified locations were then retained per query. Across the
71 true present queries, top-1 had 33 locations within 12 pixels; the saved
candidate lists contained a location within 12 pixels for 37. This shows some
ranking rather than search failures, but most misses remain retrieval failures.

An experimental geometric assignment initialized from the top-1 pattern fit
then assigned patches and constellation nodes one-to-one using their alternate
locations. On all three training scenes it scored **0.444080**, below 0.454882;
the winning names stayed the same, while a figure point moved to a worse local
match. A sweep of photometric penalties from 0 to 1000 did not alter the final
score. The experiment is opt-in as `--joint-alternatives` and is not the
default. Its pose initialization inherits the top-1 fit, so it cannot escape
an incorrect seed.

The highest measured score remains 0.454882 on three development scenes; these
tests do not approach 0.9. The next structural work is improved candidate
coverage and a joint model that can generate poses from alternatives instead
of assuming its top-1 pose is correct.

## Reproduce in PowerShell

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m constellation.cleaning --output outputs/cleaning-e2e
.\.venv\Scripts\python.exe -m constellation audit --no-preprocess-probe --output outputs/cleaning-e2e/audit
foreach ($cleaningMode in @('none','gamma','bilateral','clahe')) {
    .\.venv\Scripts\python.exe -m constellation evaluate --preprocess $cleaningMode --output "outputs/cleaning-e2e/eval-$cleaningMode"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $cleaningMode" }
}
.\.venv\Scripts\python.exe -m constellation predict --preprocess none --output outputs/cleaning-e2e/prediction
```

To reproduce the follow-up methods:

```powershell
foreach ($sigma in @(4,10,14,20,28)) {
    .\.venv\Scripts\python.exe -m constellation evaluate --background-sigma $sigma --output "outputs/cloud-sigma-$sigma"
    if ($LASTEXITCODE -ne 0) { throw "Cloud scale failed: $sigma" }
}
.\.venv\Scripts\python.exe -m constellation evaluate --joint-alternatives --output outputs/joint-alternatives
.\.venv\Scripts\python.exe -m constellation.alternative_sweep outputs/multilocation-candidates/evaluate-patches.json --output outputs/alternative-sweep.json
```

Choose a new output directory to preserve earlier experiments. Each directory
contains per-patch predictions and JSON reports. The final CSV is generated only
after the full validation set passes the existing submission contract checks.
Nothing is submitted to Kaggle automatically.

Completed run: 26 tests passed; 19 normalized archives were independently
checked for shape, finite values, and all 784 query IDs. Validation inference
finished in 294.3 seconds on this device. The generated
`outputs/cleaning-e2e/prediction/submission.csv` passed an independent check for
all 16 scenes, all 668 issued queries, exact sample-schema columns, valid class
names, and in-frame coordinates. A final hash comparison again confirmed all
853 source files unchanged. Validation accuracy is unknown without labels.
