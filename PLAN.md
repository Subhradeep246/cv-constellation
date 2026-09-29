# Local implementation plan

## Objective
Build a reproducible program that discovers scenes from their files, localizes or rejects each query patch, matches one of the supplied constellation patterns, and writes the required CSV. Run on this Windows device (32 logical CPUs, NVIDIA RTX 5070 Ti Laptop, 12 GB VRAM). No scene-specific predictions or validation-label assumptions.

## Milestones
1. **Foundation:** isolated environment, robust dataset/CSV loaders, documented local metric, unit tests, hardware and data checks.
2. **Localization:** multi-scale point-source candidates, rotation-invariant polar descriptors, top-K retrieval, angular correlation and photometric verification. Record confidence, ambiguity, and candidate recall.
3. **Identification:** extract template nodes from RGBA artwork; compare partial point sets under similarity/reflection and affine refinement. Test this stage independently with oracle training coordinates.
4. **Validation:** score the actual predictions for every training scene, save per-patch diagnostics, time/memory observations, and a submission-format smoke test. Training scores are development results, not unseen-scene estimates.
5. **Improve empirically:** scene-held-out threshold calibration, source-candidate recall, independent SIFT proposals, affine geometric hashing if required. Consider GPU descriptors only after a measured CPU bottleneck or recall failure.

## Acceptance criteria for this first implementation
- One command runs training evaluation; another predicts arbitrary scene folders using the supplied sample CSV as a schema.
- Reuses computations within a scene, caps candidate work, and supports a small smoke-test mode.
- Tests cover parsing, coordinate convention, empty predictions, localization ramp, and one-to-one geometry matching.
- Reports measured results without claiming an unmeasured leaderboard score.
- Writes only local outputs. No automatic Kaggle submission.

## Open scientific questions
- Is affine registration sufficient to map schematic reference nodes to the observed stars? Oracle-location evaluation will test this.
- How much appearance survives the patch degradation? Measure retrieval and verification separately.
- The brief says geometry uses all present predictions regardless of `m`; the local scorer follows that description, but no official scoring code has been supplied.
- The brief does not specify all tie/empty-class conventions. They must be documented rather than presented as an exact official evaluator.

## First implementation completed
- [x] Environment and dependency lock; dataset/CSV handling; local metric.
- [x] CPU polar descriptor retrieval, RootSIFT proposals, direct registration and optional streamed dense fallback.
- [x] RGBA node extraction and affine-invariant quad matching with unique point assignment.
- [x] Complete fast and dense training benchmarks, oracle probes, and threshold-only held-out analysis.
- [x] Full local validation draft and independent schema/coordinate verification.
- [x] Eleven passing tests, including inference without training labels, plus run documentation.

The fast development baseline scores 0.454882 in 26.5 seconds on the three training scenes. The dense experiment scores 0.407206 in 235.9 seconds. These are development measurements, not official leaderboard results. See `BENCHMARKS.md` for failures, evidence, and the next experiments. The pipeline is runnable; optimization toward a competition-winning solution remains open.
