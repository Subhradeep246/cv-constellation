"""Bounded CPU patch retrieval with rotation/scale-aware verification.

All coordinates are (column, row). Retrieval retains alternatives; presence
is decided only after direct photometric verification, never from filenames.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .geometry import ransac_similarity, transform_points


@dataclass
class Match:
    x: float
    y: float
    score: float
    second_score: float
    margin: float
    scale: float
    angle: float
    present: bool = False
    method: str = "polar"
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class LocalizerConfig:
    max_sources: int = 35000
    candidates_per_scale: int = 60
    scales: tuple[float, ...] = (.65, .8, .95, 1.1, 1.3, 1.55)
    refine_candidates: int = 8
    threshold: float = .80
    minimum_margin: float = 0.0
    sift: bool = True
    harris: bool = True
    ransac_trials: int = 80
    grid_step: int = 0
    background_sigma: float = 7.0
    foreground_sigma: float = .6
    descriptor_mode: str = 'magnitude'
    descriptor_whitening: bool = False


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values = values - values.mean(axis=-1, keepdims=True)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


class Localizer:
    def __init__(self, config: LocalizerConfig):
        self.config = config
        self.radii = np.linspace(2, 15, 14, dtype=np.float32)
        self.angles = np.arange(64, dtype=np.float32) * (2 * np.pi / 64)
        self.offsets = np.stack([self.radii[:, None] * np.cos(self.angles),
                                 self.radii[:, None] * np.sin(self.angles)], axis=-1)
        # The polar grid oversamples the centre; compensate by annulus area.
        self.weights = np.sqrt(self.radii)[:, None]
        self.sift = cv2.SIFT_create(nfeatures=40000, contrastThreshold=.012, edgeThreshold=14)
        self.patch_sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=.004, edgeThreshold=15)

    @staticmethod
    def filtered(image: np.ndarray, background_sigma: float = 7.0,
                 foreground_sigma: float = .6) -> np.ndarray:
        # Difference-of-Gaussians only. Extra CLAHE, gamma, or denoising is
        # intentionally omitted; constellation.audit.probe_preprocess measures it.
        arr = image.astype(np.float32)
        return (cv2.GaussianBlur(arr, (0, 0), foreground_sigma)
                - cv2.GaussianBlur(arr, (0, 0), background_sigma))

    def detect(self, image: np.ndarray) -> np.ndarray:
        arr = image.astype(np.float32)
        fine = cv2.GaussianBlur(arr, (0, 0), .9)
        response = fine - cv2.GaussianBlur(arr, (0, 0), 3.2)
        maxima = (response >= cv2.dilate(response, np.ones((5, 5), np.uint8))) & (response > 1.5)
        maxima[:22] = maxima[-22:] = False
        maxima[:, :22] = maxima[:, -22:] = False
        y, x = np.nonzero(maxima)
        strength = response[y, x]
        selected = np.argsort(strength)[-self.config.max_sources:]
        # Quadratic peak offsets reduce sensitivity of polar descriptors.
        x, y = x[selected], y[selected]
        c = fine[y, x]
        dx = .5 * (fine[y, x-1] - fine[y, x+1]) / np.minimum(fine[y, x-1] - 2*c + fine[y, x+1], -1e-5)
        dy = .5 * (fine[y-1, x] - fine[y+1, x]) / np.minimum(fine[y-1, x] - 2*c + fine[y+1, x], -1e-5)
        blobs = np.stack([x + np.clip(dx, -.75, .75), y + np.clip(dy, -.75, .75)], axis=1).astype(np.float32)
        if not self.config.harris:
            return blobs
        return self._with_harris(arr, blobs)

    def _with_harris(self, image: np.ndarray, blobs: np.ndarray) -> np.ndarray:
        """Harris corners (both eigenvalues large) supplement DoG blob peaks."""
        response = cv2.cornerHarris(image.astype(np.float32), 2, 3, 0.04)
        peaks = (response >= cv2.dilate(response, np.ones((5, 5), np.uint8))) & (
            response > 1e-5 * float(max(response.max(), 1e-12)))
        peaks[:22] = peaks[-22:] = False
        peaks[:, :22] = peaks[:, -22:] = False
        y, x = np.nonzero(peaks)
        if not len(x):
            return blobs
        order = np.argsort(response[y, x])[::-1][:4000]
        corners = np.stack([x[order], y[order]], axis=1).astype(np.float32)
        if not len(blobs):
            return corners
        nearest, _ = cKDTree(blobs).query(corners, k=1)
        extra = corners[nearest > 4]
        if not len(extra):
            return blobs
        return np.concatenate([blobs, extra])

    def sample(self, image: np.ndarray, centers: np.ndarray, scales: np.ndarray | float) -> np.ndarray:
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        scales = np.broadcast_to(np.asarray(scales, dtype=np.float32), (len(centers),))
        out = np.empty((len(centers), len(self.radii), len(self.angles)), dtype=np.float32)
        # OpenCV remap dimensions must remain below 32767.
        for start in range(0, len(centers), 256):
            stop = min(start + 256, len(centers))
            xy = centers[start:stop, None, None, :] + self.offsets[None] * scales[start:stop, None, None, None]
            values = cv2.remap(image, xy[..., 0].reshape(-1, 64), xy[..., 1].reshape(-1, 64),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            out[start:stop] = values.reshape(stop-start, len(self.radii), 64)
        return out

    def descriptor(self, polar: np.ndarray) -> np.ndarray:
        polar = polar - polar.mean(axis=(-2, -1), keepdims=True)
        fourier = np.fft.rfft(polar, axis=-1)[..., :9]
        spectrum = np.abs(fourier)
        # Suppress the uninformative circular central-star profile. Retain
        # a little radial information to distinguish scale and background.
        spectrum[..., 0] *= .12
        spectrum = spectrum * self.weights
        # Magnitude is invariant to circular angular shifts (rotation).
        flat = spectrum.reshape(len(polar), -1).astype(np.float32)
        if self.config.descriptor_mode == 'crossphase':
            # A rotation multiplies all radii of harmonic k by the same phase.
            # Referencing each harmonic to its radial sum cancels that phase,
            # preserving relative azimuth between the rings.
            reference = np.sum(fourier[:, :, 1:9] * self.weights[None], axis=1)
            unit = np.conj(reference) / np.maximum(np.abs(reference), 1e-6)
            aligned = fourier[:, :, 1:9] * unit[:, None, :] * self.weights[None]
            phase_features = np.concatenate([aligned.real, aligned.imag], axis=-1)
            phase_features = phase_features.reshape(len(polar), -1).astype(np.float32)
            phase_features /= np.maximum(np.linalg.norm(phase_features, axis=1, keepdims=True), 1e-8)
            flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
            flat = np.concatenate([flat, 1.5 * phase_features], axis=1)
        return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)

    def rotation_scores(self, query: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = (query - .95*query.mean(axis=-1,keepdims=True) - .05*query.mean()) * self.weights
        c = (candidate - .95*candidate.mean(axis=-1,keepdims=True) - .05*candidate.mean(axis=(-2,-1),keepdims=True)) * self.weights
        cross = np.sum(np.fft.rfft(c, axis=-1) * np.conj(np.fft.rfft(q, axis=-1)), axis=1)
        correlations = np.fft.irfft(cross, n=64, axis=-1)
        correlations /= np.maximum(np.linalg.norm(c, axis=(1, 2))[:, None] * np.linalg.norm(q), 1e-8)
        angles = correlations.argmax(axis=1)
        return correlations[np.arange(len(candidate)), angles], angles * (2*np.pi/64)

    @staticmethod
    def rootsift(descriptors: np.ndarray) -> np.ndarray:
        return np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)

    def sift_align(self, patch: np.ndarray, keypoints, matcher) -> tuple[float, float, float, int] | None:
        """SIFT correspondences → RANSAC similarity. This is the alignment, not a spare guess."""
        if matcher is None or keypoints is None:
            return None
        enlarged = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        kp, desc = self.patch_sift.detectAndCompute(enlarged, None)
        if desc is None or len(desc) < 2:
            return None
        pairs = matcher.knnMatch(self.rootsift(desc), k=2)
        center = (np.array(patch.shape[::-1], dtype=float) - 1) / 2
        source, destination = [], []
        for group in pairs:
            if not group:
                continue
            if len(group) >= 2 and group[0].distance >= 0.9 * group[1].distance:
                continue
            a, b = kp[group[0].queryIdx], keypoints[group[0].trainIdx]
            patch_xy = (np.array(a.pt, dtype=float) + .5) / 4 - .5
            source.append(patch_xy)
            destination.append(np.array(b.pt, dtype=float))
        if len(source) < 3:
            return None
        consensus = ransac_similarity(
            np.asarray(source), np.asarray(destination),
            threshold=8.0, trials=max(self.config.ransac_trials, 120),
            rng=np.random.default_rng(17))
        # Two points define a similarity; a third inlier is the consensus check.
        if consensus is None or consensus.n_inliers < 3:
            return None
        mapped = transform_points(consensus.model, center[None])[0]
        scale = float(np.hypot(consensus.model[0, 0], consensus.model[0, 1]))
        if not .45 < scale < 2.1 or not np.isfinite(mapped).all():
            return None
        return float(mapped[0]), float(mapped[1]), scale, int(consensus.n_inliers)

    def sift_match_proposals(self, patch: np.ndarray, keypoints, matcher) -> list[tuple[float, float, float]]:
        """Per-match similarity hypotheses; used only if RANSAC is rejected."""
        if matcher is None:
            return []
        enlarged = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        kp, desc = self.patch_sift.detectAndCompute(enlarged, None)
        if desc is None:
            return []
        proposals = []
        center = (np.array(patch.shape[::-1], dtype=float) - 1) / 2
        for group in matcher.knnMatch(self.rootsift(desc), k=3):
            for match in group:
                a, b = kp[match.queryIdx], keypoints[match.trainIdx]
                scale = 4 * b.size / max(a.size, 1e-6)
                if not .45 < scale < 2.1:
                    continue
                theta = np.deg2rad(b.angle - a.angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                patch_xy = (np.array(a.pt, dtype=float) + .5) / 4 - .5
                location = np.array(b.pt) + scale * rotation @ (center - patch_xy)
                proposals.append((float(location[0]), float(location[1]), float(scale)))
        return proposals

    def verify(self, image: np.ndarray, patch: np.ndarray, proposals: np.ndarray,
               method: str = "polar") -> Match:
        center = (np.array(patch.shape[::-1], dtype=float)-1)/2
        filtered = self.filtered(patch, self.config.background_sigma,
                                 self.config.foreground_sigma)
        query = self.sample(filtered, center[None], 1)[0]
        if not len(proposals) or np.std(query) < 1e-4:
            return Match(0, 0, 0, 0, 0, 1, 0, method=method)
        polar = self.sample(image, proposals[:, :2], proposals[:, 2])
        scores, angles = self.rotation_scores(query, polar)
        order = np.argsort(scores)[::-1]
        selected = []
        for i in order:
            if all(np.linalg.norm(proposals[i, :2] - proposals[j, :2]) > 10 for j in selected):
                selected.append(i)
            if len(selected) >= self.config.refine_candidates:
                break
        # Optimize on the actual square patch inside its inscribed circle.
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]].astype(np.float32)
        xx, yy = xx-center[0], yy-center[1]
        mask = (xx*xx + yy*yy) <= 15**2
        u, v = xx[mask], yy[mask]
        radial_bin = np.rint(np.hypot(u,v)).astype(int)
        radial_count = np.bincount(radial_bin)
        def whiten(values):
            radial = np.bincount(radial_bin,weights=values,minlength=len(radial_count))/np.maximum(radial_count,1)
            values = values - .95*radial[radial_bin]
            return normalized_rows(values[None])[0]
        q = whiten(filtered[mask])
        def objective(params):
            x, y, scale, angle = params
            co, si = np.cos(angle), np.sin(angle)
            mx = (x + scale*(co*u-si*v)).astype(np.float32)[None]
            my = (y + scale*(si*u+co*v)).astype(np.float32)[None]
            crop = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)[0]
            return -float(whiten(crop) @ q)
        results = []
        for i in selected:
            x, y, scale = proposals[i]
            initial = [float(x), float(y), float(scale), float(angles[i])]
            bounds = [(max(0, x-3), min(image.shape[1]-1, x+3)),
                      (max(0, y-3), min(image.shape[0]-1, y+3)),
                      (max(.4, scale*.86), min(2.3, scale*1.16)),
                      (angles[i]-.18, angles[i]+.18)]
            if any(lo >= hi for lo, hi in bounds):
                continue
            opt = minimize(objective, initial, method="Powell", bounds=bounds,
                           options={"maxiter":20, "xtol":.025, "ftol":.0005})
            # Retain the initial estimate if a local optimization got worse.
            initial_score = -objective(initial)
            score, params = (-float(opt.fun), opt.x) if -opt.fun > initial_score else (initial_score, initial)
            results.append((score, params))
        if not results:
            return Match(0, 0, 0, 0, 0, 1, 0, method=method)
        results.sort(key=lambda item: item[0], reverse=True)
        score, (x, y, scale, angle) = results[0]
        # Retain spatially distinct verified solutions so global figure geometry
        # can arbitrate duplicated stellar neighborhoods later.
        locations = []
        for value, params in results:
            rx, ry, rs, ra = params
            if all(np.hypot(rx - item["x"], ry - item["y"]) > 10 for item in locations):
                locations.append(dict(x=float(rx), y=float(ry), score=float(value),
                                      scale=float(rs), angle=float(ra)))
            if len(locations) >= self.config.refine_candidates:
                break
        second = locations[1]["score"] if len(locations) > 1 else 0.0
        return Match(float(x), float(y), float(score), float(second), float(score-second), float(scale),
                     float(angle), bool(score >= self.config.threshold and score-second >= self.config.minimum_margin),
                     method, locations)

    def match_scene(self, sky: np.ndarray, patches: list[np.ndarray], log=print) -> tuple[list[Match], dict]:
        cv2.setRNGSeed(17)
        started = time.perf_counter()
        sources = self.detect(sky)
        log(f"  {len(sources):,} source candidates")
        if self.config.grid_step > 0:
            y, x = np.mgrid[16:sky.shape[0]-16:self.config.grid_step, 16:sky.shape[1]-16:self.config.grid_step]
            sources = np.concatenate([sources,np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)])
            log(f"  {len(sources):,} total positions including the regular-grid fallback")
        filtered = self.filtered(sky, self.config.background_sigma,
                                 self.config.foreground_sigma)
        alignments: list[tuple[float, float, float, int] | None] = [None] * len(patches)
        sift_extra: list[list[tuple[float, float, float]]] = [[] for _ in patches]
        if self.config.sift:
            kp, desc = self.sift.detectAndCompute(sky, None)
            matcher = None
            if desc is not None and len(desc) >= 3:
                matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
                matcher.add([self.rootsift(desc)])
                matcher.train()
            for j, patch in enumerate(patches):
                alignments[j] = self.sift_align(patch, kp, matcher)
                sift_extra[j] = self.sift_match_proposals(patch, kp, matcher)
            log(f"  SIFT+RANSAC aligned {sum(p is not None for p in alignments)}/{len(patches)} patches "
                f"from {0 if kp is None else len(kp):,} sky keypoints")
        matches: list[Match | None] = [None] * len(patches)
        proposal_counts = [0] * len(patches)
        ransac_kept = 0
        for j, pose in enumerate(alignments):
            if pose is None:
                continue
            x, y, scale, _ = pose
            candidates = np.array([[x, y, scale], [x, y, max(.45, scale*.9)], [x, y, min(2.1, scale*1.1)]],
                                  dtype=np.float32)
            good = ((candidates[:, 0] >= 4) & (candidates[:, 0] < sky.shape[1]-4) &
                    (candidates[:, 1] >= 4) & (candidates[:, 1] < sky.shape[0]-4))
            proposal_counts[j] = int(good.sum())
            match = self.verify(filtered, patches[j], candidates[good], method="ransac")
            if match.present:
                matches[j] = match
                ransac_kept += 1
        fallback = [j for j, match in enumerate(matches) if match is None]
        log(f"  RANSAC accepted {ransac_kept}/{len(patches)}; polar fallback {len(fallback)}")
        if fallback:
            subset = [patches[j] for j in fallback]
            queries = np.stack([self.sample(self.filtered(p, self.config.background_sigma,
                                                         self.config.foreground_sigma),
                                            ((np.array(p.shape[::-1])-1)/2)[None], 1)[0]
                                for p in subset])
            qdesc = self.descriptor(queries)
            pooled = [[] for _ in subset]
            for scale in self.config.scales:
                count = min(self.config.candidates_per_scale, len(sources))
                if not count:
                    continue
                if self.config.descriptor_whitening:
                    sample_ids = np.linspace(0, len(sources)-1,
                                             min(2048, len(sources)), dtype=int)
                    representative = self.descriptor(
                        self.sample(filtered, sources[sample_ids], scale))
                    center = representative.mean(axis=0)
                    spread = np.sqrt(representative.var(axis=0) + .02 ** 2)
                    search_queries = normalized_rows((qdesc - center) / spread)
                else:
                    search_queries = qdesc
                best_scores=np.full((len(subset),count),-np.inf,dtype=np.float32)
                best_indices=np.zeros((len(subset),count),dtype=int)
                for start in range(0,len(sources),4096):
                    positions=sources[start:start+4096]
                    desc=self.descriptor(self.sample(filtered,positions,scale))
                    if self.config.descriptor_whitening:
                        desc = normalized_rows((desc - center) / spread)
                    similarities=search_queries@desc.T
                    joined=np.concatenate([best_scores,similarities],axis=1)
                    indices=np.concatenate([best_indices,np.broadcast_to(np.arange(start,start+len(positions)),similarities.shape)],axis=1)
                    top=np.argpartition(joined,-count,axis=1)[:,-count:]
                    best_scores=np.take_along_axis(joined,top,axis=1)
                    best_indices=np.take_along_axis(indices,top,axis=1)
                for i in range(len(subset)):
                    pooled[i].append(np.column_stack([sources[best_indices[i]], np.full(count, scale)]))
                log(f"  polar fallback scale {scale:.2f} ({len(fallback)} patches)")
            for i, j in enumerate(fallback):
                parts = pooled[i][:]
                if sift_extra[j]:
                    parts.append(np.array(sift_extra[j], dtype=np.float32))
                candidates = np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 3))
                good = ((candidates[:, 0] >= 4) & (candidates[:, 0] < sky.shape[1]-4) &
                        (candidates[:, 1] >= 4) & (candidates[:, 1] < sky.shape[0]-4))
                proposal_counts[j] = int(good.sum())
                matches[j] = self.verify(filtered, patches[j], candidates[good], method="polar")
                if (i+1) % 10 == 0 or i+1 == len(fallback):
                    log(f"  verified fallback {i+1}/{len(fallback)}")
        for j, match in enumerate(matches):
            if match is None:
                matches[j] = Match(0, 0, 0, 0, 0, 1, 0, method="polar")
        return matches, {"source_count":len(sources), "seconds":time.perf_counter()-started,
                         "proposals_per_patch": proposal_counts,
                         "ransac_aligned": sum(p is not None for p in alignments),
                         "ransac_accepted": ransac_kept,
                         "polar_fallback": len(fallback)}
