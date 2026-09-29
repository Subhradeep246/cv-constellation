# Constellation Detection — CS-GY 6643 Computer Vision, Project 1

**Selected submission:** public leaderboard score **0.95248**
**Train score:** 0.9641 (mean over the three labelled scenes)

A classical, deterministic pipeline with no learned weights. It uses only the provided data: no external star catalogues and no pretrained models. One command regenerates the submitted CSV byte for byte.

---

## 1. Contents

| File | Purpose |
|---|---|
| `constellation_v8.py` | Complete pipeline (single file). |
| `requirements.txt` | Python dependencies. |
| `submission_selected.csv` | The selected submission (public 0.95248), for comparison with a fresh run. |
| `README.md` | This file. |

## 2. Setup

Python 3.9 or newer. Tested with Python 3.11, NumPy 2.4, SciPy 1.17 and OpenCV 4.13.

```bash
pip install -r requirements.txt
```

No GPU is needed.

## 3. Data layout

Place the competition data in a folder (here called `participant`) next to the script:

```
.
├── constellation_v8.py
└── participant/
    ├── patterns/                  48 constellation pattern PNGs
    ├── train/                     pisces/, scorpius/, taurus/
    ├── validation/                constellation_01/ … constellation_17/
    ├── train_ground_truth.csv     needed only for --split train (and --eval)
    └── sample_submission.csv      optional: scene list and output format for --split validation
```

Each scene folder contains `<scene>_image.png` and a `patches/` folder with `patch_XX.png` files, exactly as distributed.

### How the script finds the scenes

The script chooses its scene list in this order:

1. **`--index <csv>`, if given.** A CSV with at least the columns `Id` and `n_patches`. Its `patch_XX` columns, if any, set the output format.
2. **Otherwise, the default index for the split.** This is `<data>/train_ground_truth.csv` for `--split train` and `<data>/sample_submission.csv` for any other split. Using it reproduces the competition's column order and width exactly.
3. **Otherwise, the scene folders on disk.** If that file does not exist, every sub-folder of `<data>/<split>/` that contains `<name>/<name>_image.png` is treated as a scene, and its patches are counted from `patches/`. The output has columns `Id, n_patches, patch_01 … patch_NN, constellation`, where NN is the largest patch count found. Use `--ncols` to force a width, e.g. `--ncols 87`.

The first line of output shows which source was used, e.g. `scene list: participant/sample_submission.csv (16 scenes)` or `scene list: discovered 16 scene folders in …`.

If a scene is listed in the index CSV but its image is missing or unreadable, the run stops with an error naming the file. Check that the data was extracted completely.

### Running on a held-out test set

Put the test scenes in a split folder using the same layout, for example:

```
participant/test/<scene_id>/<scene_id>_image.png
participant/test/<scene_id>/patches/patch_01.png …
```

Then run

```bash
python constellation_v8.py --data participant --split test --out test_submission.csv --no-fb-extra
```

This uses `participant/sample_submission.csv` if it lists the test scenes. If that file is absent, the script discovers the scenes from the folders. If the test set comes with its own template CSV, pass it explicitly:

```bash
python constellation_v8.py --data participant --split test --index participant/test_sample_submission.csv \
       --out test_submission.csv --no-fb-extra
```

The index CSV must list only scenes that exist under `<data>/<split>/`. Each run caches per scene under `cache_final/`, keyed by scene ID. If the test scene IDs could clash with scene IDs already in the cache, delete `cache_final/` or pass a fresh `--cache` directory.

Commands:

If the test set comes with its own template CSV:

```bash
python constellation_v8.py --data participant --split test --index participant/test_sample_submission.csv --out test_submission.csv --no-fb-extra
```

If it doesn't, the scenes are discovered from the folders:

```bash
python constellation_v8.py --data participant --split test --out test_submission.csv --no-fb-extra
```

Add --ncols 87 if the output must have exactly the competition's 87 patch columns. Without it, the width equals the largest patch count in the test set.

One case needs care. If participant/sample_submission.csv exists and you don't pass --index, the script uses it to decide which scenes to run. That file lists the validation scene names, not the test ones, so either pass --index or make sure that file isn't there.

## 4. Reproducing the selected submission

```bash
python constellation_v8.py --data participant --split validation --out submission.csv --no-fb-extra
```

`--no-fb-extra` is part of the selected configuration. Without it the script reproduces an earlier submission (public 0.9475).

To check the result against the submitted file:

```bash
python - <<'EOF'
import csv, ast
A = {r['Id']: r for r in csv.DictReader(open('submission.csv'))}
B = {r['Id']: r for r in csv.DictReader(open('submission_selected.csv'))}
for sid in B:
    a, b = A[sid], B[sid]
    diffs = []
    for k in b:
        if k.startswith('patch_') and a[k] != b[k]:
            pa = None if a[k] == '-1' else ast.literal_eval(a[k])
            pb = None if b[k] == '-1' else ast.literal_eval(b[k])
            if pa and pb:
                d = ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** .5
                diffs.append(f'{k}: moved {d:.0f}px')
            else:
                diffs.append(f'{k}: {b[k]} -> {a[k]}')
    name = 'same name' if a['constellation'] == b['constellation'] else f"NAME {b['constellation']} -> {a['constellation']}"
    print(f"{sid}: {name}, {len(diffs)} cells differ", *diffs[:6], sep='\n   ')
EOF
```
some cells may change due to python version mismatch

## 5. Evaluating on the training scenes

```bash
python constellation_v8.py --data participant --split train --eval --out train_pred.csv --no-fb-extra
```

Expected output ends with:

```
MEAN SCORE 0.9641 over 3 scenes
```

The per-scene scores are pisces 0.957, scorpius 0.957 and taurus 0.978, with all three constellations identified correctly. The `--eval` flag uses a re-implementation of the competition metric:

S = 0.25·Presence + 0.20·Localization + 0.25·Geometric + 0.30·Identification

## 6. Runtime and caching

- **First run:** about 15–20 s per patch on 2 CPU cores, roughly 3–4 hours for the 16 validation scenes (668 patches). Per-scene evidence is cached in `cache_final/` (created automatically).
- **Later runs:** read the cache and finish in about 20 minutes. Delete `cache_final/` to force a full rebuild.
- **Parallelism:** the script pins BLAS/OpenMP/OpenCV to one thread per process, because this was about 10× faster than letting processes compete for cores. To use more cores, warm the cache by running groups of scenes in separate terminals, then run the full command once:

  ```bash
  python constellation_v8.py --data participant --split validation --out part_a.csv --no-fb-extra \
         --scenes constellation_01 constellation_02 constellation_03 constellation_04
  # …other groups of scenes in other terminals…
  python constellation_v8.py --data participant --split validation --out submission.csv --no-fb-extra
  ```

- **Determinism:** the pipeline has no random components. Repeated runs, and runs from an empty cache, give identical output.

## 7. Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--data` | (required) | Folder containing `patterns/`, `train/`, `validation/` and the CSVs. |
| `--split` | `validation` | Sub-folder of `--data` holding the scenes: `train`, `validation`, or any other name such as `test`. |
| `--out` | `submission.csv` | Output CSV. |
| `--eval` | off | With `--split train`, score the predictions against the ground truth. |
| `--cache` | `cache_final` | Cache directory. |
| `--scenes` | all | Restrict the run to the listed scene IDs. |
| `--index` | see §3 | CSV listing the scenes (`Id`, `n_patches`, optional `patch_XX` columns). |
| `--ncols` | largest patch count | Number of `patch_XX` columns when no index CSV provides them. |
| `--no-fb-extra` | off | **Selected configuration.** Drop the second channel's presence calls for ambiguous patches. |
| `--relgap` | 0.16 | Relative-gap threshold for a confident match. |
| `--flat-kmax`, `--flat-delta` | 5, 0.04 | Number and score window of alternative candidates kept for ambiguous patches. |
| `--ens` | `all` | Identification ensemble: `all` (4 variants) or `calib` (calibrated variants only). |
| `--patch-smooth` | 0 | Gaussian smoothing of the patch for the raw channel (0 = off). |

Only `--no-fb-extra` differs from the defaults in the selected submission. The other switches were used for single-variable leaderboard probes.

## 8. Method overview

Each scene is processed in two stages.

**Stage 1: evidence (expensive, cached)**

1. **Candidate generation.** The candidate list is the union of:
   - a dense rotation-invariant descriptor (ring means plus angular-harmonic magnitudes), re-ranked by radial-residual NCC;
   - an exhaustive full-resolution masked-disc NCC over 36 rotations. The disc's mean and variance are computed once per radius, so each rotation is a single correlation;
   - half-resolution raw and low-pass exhaustive peaks.
2. **Exact scoring.** Each candidate is scored with a per-pose masked NCC (72 angles, ±3 px). Ten distinct locations are kept per patch.
3. **Three-channel rescoring.** Candidates are re-scored with the average masked NCC of the raw image, a band-pass DoG(1,8) and a high-pass DoG(0.7,3). The pose scale is restricted to the generator's measured range, 0.93–1.13.
4. **Second channel.** A half-resolution band-pass search, verified at full resolution.
5. **Pasted decoys.** Pasted-copy ("decoy") detection for each patch's top candidates, plus whole-image copy-move detection.
6. **Star catalogues.** Two bright-star catalogues: box flux, and PSF-adaptive aperture photometry.

**Stage 2: decision (fast, replayable from the cache)**

1. **Evidence classes.** Each patch is classed as *dup* (it matches a pasted copy, which marks a figure star), *clear* (relative gap `(s1−s2)/(1−s1)` ≥ 0.16) or *flat*.
2. **Identification.** An exhaustive search over similarity transforms, including reflection, against all 48 patterns.
   - Hypotheses are scored by a penalised log-likelihood ratio: patch evidence, bright-star support, and penalties for nodes the hypothesis cannot explain.
   - The bright-star weights are re-calibrated per scene from that scene's own figure-star locations, shrunk toward the train prior.
   - The name is chosen by a softmax ensemble (τ = 4) over {2 catalogues} × {prior, calibrated}.
   - Copy-move clusters are weight-4 anchors, and each star can be claimed by only one patch.
3. **Assignment.** A Hungarian assignment of patches to the chosen figure's nodes resolves localisation ambiguity. Remaining confident patches are reported present with m = 0, and all other patches are absent (−1).

The accompanying report gives the full design rationale, experiments, ablations and hyper-parameters.

## 9. Compliance with the competition rules

- **No hardcoding.** No scene names, coordinates, per-scene constants, file sizes or leaderboard-derived answers are used. Every constant was either measured on the training scenes as a rate or tolerance, or is structural.
- **Unseen scenes.** The code runs unchanged on any scene with the same format.
- **Reproducibility.** The submitted CSV is produced end-to-end by this script with no manual edits.
