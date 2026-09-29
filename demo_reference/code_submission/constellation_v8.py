#!/usr/bin/env python3
"""
Constellation Detection (CS-GY 6643) - final pipeline (v8 = v7 + probe switches; defaults reproduce v7).

    python constellation_final.py --data participant --split validation --out submission.csv
    python constellation_final.py --data participant --split train --eval --out train_pred.csv

Two phases per scene
  1. EVIDENCE (expensive, cached to <cache>/evid/<scene>.pkl)
     - candidate locations per patch = union of
         (a) dense rotation-invariant ring/harmonic descriptor shortlist (+ radial-residual NCC re-rank)
         (b) NEW: exhaustive full-resolution masked-disk NCC over 36 rotations x 2 scales
             (plain correlation with a zero-mean disk template; disk variance computed once per radius)
     - every candidate re-scored with per-pose masked NCC (72 angles x 7 scales, +-3 px) -> 10 distinct peaks
     - annulus NCC at the best pose, half-resolution band-pass second channel with full-res verification
     - duplicate (pasted decoy disc) detection, whole-image copy-move detection
     - two bright-star catalogues: box-flux (v1) and PSF-adaptive aperture photometry (v2)
  2. DECISION (fast, replayable)
     - evidence classes dup / clear / flat, second-channel presence rules
     - similarity-transform (with reflection) identification against the 48 patterns
     - NEW: bright-star evidence weights re-calibrated per scene from its own duplicate
       (figure-star) locations, shrunk towards the train prior
     - NEW: identification = softmax ensemble over {v1, v2 catalogue} x {train-prior, calibrated}
     - Hungarian patch<->node assignment, unassigned confident patches present with m=0
  v5 additions (see the block above run_aug):
     - candidates enlarged with half-resolution raw / low-pass exhaustive peaks
     - every candidate re-scored with the average of raw, band-pass and high-pass masked NCC
     - pose scale restricted to the generator's measured range 0.93-1.13
     - duplicate evidence taken from both the combined and the high-pass rankings
  v6: whole-image copy-move clusters are weight-4 anchors (duplicated patches stay 10) and no longer enter
      the duplicate-coverage and unexplained-duplicate terms
  v7: one star per patch for duplicate pairs; copy-move pairs already explained by a duplicated patch dropped

Deterministic; no per-scene constants; every scene gets a name.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('CV_THREADS', '1')
import os, sys, csv, glob, ast, time, argparse, pickle
import numpy as np, cv2
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

cv2.setNumThreads(int(os.environ.get("CV_THREADS", max(1, os.cpu_count() or 1))))
import pickle, hashlib
EXH_MODE = os.environ.get('EXH_MODE', 'union')   # union | only | off
EXH_N = int(os.environ.get('EXH_N', '30'))
CACHE = os.environ.get('EXH_CACHE', 'cache')

# ----------------------------------------------------------------------------- IO
def load_scene(d):
    name = os.path.basename(os.path.normpath(d))
    im = cv2.imread(os.path.join(d, f'{name}_image.png'), cv2.IMREAD_GRAYSCALE)
    if im is None: raise FileNotFoundError(f'cannot read sky image {d}/{name}_image.png - the scene is listed in the index CSV '
                                           f'but its folder or image is missing, misnamed or unreadable')
    ps = sorted(glob.glob(os.path.join(d, 'patches', 'patch_*.png')),
                key=lambda p: int(os.path.basename(p)[6:-4]))
    return im, [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in ps]

def load_pattern(fn):
    p = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if p.ndim == 2: p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGRA)
    elif p.shape[2] == 3: p = cv2.cvtColor(p, cv2.COLOR_BGR2BGRA)
    b, g, r, a = [p[..., i].astype(np.int32) for i in range(4)]
    white = ((a > 100) & (r > 150) & (g > 150) & (b > 150)).astype(np.uint8)   # star dots
    n, lab, st, cen = cv2.connectedComponentsWithStats(white, 8)
    nodes = [cen[k] for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 2]
    if len(nodes) < 2:                                                          # fallback: blobs in alpha
        al = (a > 100).astype(np.uint8)
        op = cv2.morphologyEx(al, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(op, 8)
        nodes = [cen[k] for k in range(1, n)]
    return np.array(nodes, np.float64)

def load_patterns(d):
    return {os.path.basename(f)[:-len('_pattern.png')]: load_pattern(f)
            for f in sorted(glob.glob(os.path.join(d, '*_pattern.png')))}

def norm_pattern(N):
    z = N[:, 0] + 1j * N[:, 1]; z = z - z.mean()
    return z / np.abs(z[:, None] - z[None]).max()

# ----------------------------------------------------------------------------- warping helpers
def warp_crop(img, x, y, ang, s, sz=32):
    c = (sz - 1) / 2.; th = np.deg2rad(ang); ca, sa = np.cos(th) / s, np.sin(th) / s
    A = np.array([[ca, -sa, 0], [sa, ca, 0]], np.float64)
    A[:, 2] = np.array([x, y]) - A[:, :2] @ np.array([c, c])
    return cv2.warpAffine(img, A, (sz, sz), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)

def hp_pre(f):
    f = f.astype(np.float32); f = f - cv2.GaussianBlur(f, (0, 0), 3, borderType=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(f, (0, 0), 0.7, borderType=cv2.BORDER_REFLECT)

# ----------------------------------------------------------------------------- 1a dense descriptor
RING_EDGES = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12.0]); KH = [1, 2, 3]; STRIDE = 2
def _ring_kernels(R=12):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    rr = np.hypot(xx, yy); th = np.arctan2(yy, xx); ks = []
    for j in range(len(RING_EDGES) - 1):
        m = ((rr >= RING_EDGES[j]) & (rr < RING_EDGES[j + 1])).astype(np.float64); m /= m.sum()
        ks.append((j, 0, m))
        for k in KH: ks.append((j, k, m * np.exp(-1j * k * th)))
    disk = (rr < RING_EDGES[-1]).astype(np.float64); disk /= disk.sum()
    return ks, disk
KS, DISK = _ring_kernels()
def _filt(f, k): return cv2.filter2D(f, cv2.CV_32F, k.astype(np.float32), borderType=cv2.BORDER_REFLECT)
def dense_desc(im):
    f = im.astype(np.float32)
    mu = _filt(f, DISK); sd = np.sqrt(np.maximum(_filt(f * f, DISK) - mu * mu, 1e-2))
    s = slice(0, None, STRIDE); mu_s, sd_s = mu[s, s], sd[s, s]; feats = []
    for j, k, ker in KS:
        if k == 0: feats.append((_filt(f, ker)[s, s] - mu_s) / sd_s)
        else:
            re = _filt(f, ker.real)[s, s]; imz = _filt(f, ker.imag)[s, s]
            feats.append(np.sqrt(re * re + imz * imz) / sd_s)
    return np.stack(feats, -1).astype(np.float32)
def patch_desc(p, scale):
    p = p.astype(np.float32); c = (p.shape[0] - 1) / 2.; R = 12
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    w = cv2.remap(p, (c + xx * scale).astype(np.float32), (c + yy * scale).astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float64)
    mu = (w * DISK).sum(); sd = np.sqrt(max((w * w * DISK).sum() - mu * mu, 1e-2)); out = []
    for j, k, ker in KS:
        v = (w * ker).sum(); out.append((v - mu) / sd if k == 0 else abs(v) / sd)
    return np.array(out, np.float32)

# ----------------------------------------------------------------------------- 1b radial-residual NCC (R=11)
R1 = 11; PAD1 = 4
_y1, _x1 = np.mgrid[-R1:R1 + 1, -R1:R1 + 1].astype(np.float32)
MI1 = np.nonzero(((_x1 ** 2 + _y1 ** 2) <= R1 * R1 + 0.5).ravel())[0]; NM1 = len(MI1)
_rb = np.round(np.sqrt(_x1 ** 2 + _y1 ** 2).ravel()[MI1]).astype(int)
ONE = np.zeros((NM1, _rb.max() + 1), np.float32); ONE[np.arange(NM1), _rb] = 1; CNT = ONE.sum(0)
ANG1 = np.arange(0, 360, 5.0); SC1 = np.array([0.85, 0.95, 1.05, 1.17, 1.3, 1.45])
def templates1(p):
    c = (p.shape[0] - 1) / 2.; T = []
    for s in SC1:
        for a in ANG1:
            th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th)
            t = cv2.remap(p, (c + s * (ca * _x1 - sa * _y1)).astype(np.float32),
                          (c + s * (sa * _x1 + ca * _y1)).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()[MI1]
            T.append(t)
    T = np.array(T, np.float32).T
    m = (ONE.T @ T) / CNT[:, None]; Tr = T - ONE @ m
    return Tr / (np.linalg.norm(Tr, axis=0, keepdims=True) + 1e-6)
def match1(imf, Tr, cands, chunk=64):
    H, W = imf.shape; h = R1 + PAD1; w = 2 * PAD1 + 1; out = np.zeros((len(cands), 3), np.float32)
    for c0 in range(0, len(cands), chunk):
        cc = cands[c0:c0 + chunk]
        xs = np.clip(cc[:, 0], h, W - 1 - h).astype(int); ys = np.clip(cc[:, 1], h, H - 1 - h).astype(int)
        crops = np.array([imf[y - h:y + h + 1, x - h:x + h + 1] for x, y in zip(xs, ys)])
        win = swv(crops, (2 * R1 + 1, 2 * R1 + 1), axis=(1, 2)).reshape(len(cc), w * w, -1)[:, :, MI1]
        wr = win - ((win @ ONE) / CNT) @ ONE.T
        S = ((wr / (np.linalg.norm(wr, axis=-1, keepdims=True) + 1e-3)) @ Tr).reshape(len(cc), -1)
        j = S.argmax(1); pos = j // Tr.shape[1]; dy, dx = np.divmod(pos, w)
        out[c0:c0 + len(cc)] = np.stack([S[np.arange(len(cc)), j], xs - PAD1 + dx, ys - PAD1 + dy], 1)
    return out

# ----------------------------------------------------------------------------- 1c masked NCC (R=18)
R3 = 18
ANG3 = np.arange(0, 360, 5.); SC3 = np.array([0.85, 0.93, 1.0, 1.08, 1.17, 1.27, 1.4])
_y3, _x3 = np.mgrid[-R3:R3 + 1, -R3:R3 + 1].astype(np.float32)
MI3 = np.nonzero(((_x3 ** 2 + _y3 ** 2) <= R3 * R3 + 0.5).ravel())[0]
def templates3(p, margin=0.5):
    c = (p.shape[0] - 1) / 2.; T = []; M = []; meta = []
    for s in SC3:
        for a in ANG3:
            th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th)
            mx = (c + s * (ca * _x3 - sa * _y3)).astype(np.float32); my = (c + s * (sa * _x3 + ca * _y3)).astype(np.float32)
            valid = ((mx >= -margin) & (mx <= p.shape[1] - 1 + margin) & (my >= -margin) &
                     (my <= p.shape[0] - 1 + margin)).ravel()[MI3].astype(np.float32)
            t = cv2.remap(p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[MI3]
            n = valid.sum(); t = (t - (t * valid).sum() / n) * valid; t /= np.linalg.norm(t) + 1e-6
            T.append(t); M.append(valid); meta.append((a, s))
    T = np.array(T, np.float32).T; M = np.array(M, np.float32).T
    return T, M, M.sum(0), np.array(meta)
def match3(imf, T, M, NMk, meta, cands, pad=3, chunk=32):
    H, W = imf.shape; h = R3 + pad; w = 2 * pad + 1; out = np.zeros((len(cands), 5), np.float32)
    for c0 in range(0, len(cands), chunk):
        cc = cands[c0:c0 + chunk]
        xs = np.clip(cc[:, 0], h, W - 1 - h).astype(int); ys = np.clip(cc[:, 1], h, H - 1 - h).astype(int)
        crops = np.array([imf[y - h:y + h + 1, x - h:x + h + 1] for x, y in zip(xs, ys)])
        win = swv(crops, (2 * R3 + 1, 2 * R3 + 1), axis=(1, 2)).reshape(len(cc), w * w, -1)[:, :, MI3]
        num = win @ T; s1 = win @ M; s2 = (win * win) @ M
        S = (num / np.sqrt(np.maximum(s2 - s1 * s1 / NMk, 1e-3))).reshape(len(cc), -1)
        j = S.argmax(1); ar = np.arange(len(cc)); pos, k = np.divmod(j, T.shape[1]); dy, dx = np.divmod(pos, w)
        out[c0:c0 + len(cc)] = np.stack([S[ar, j], xs - pad + dx, ys - pad + dy, meta[k, 0], meta[k, 1]], 1)
    return out

# ----------------------------------------------------------------------------- 1d fallback band-pass channel
FB_LO, FB_HI = 1.0, 8.0          # fallback-channel band, applied IDENTICALLY to image and patch
def fb_pre(f):
    """Band-pass for the fallback channel.

    The image and the patch must get the SAME band: the image was low-passed at sigma=1 and the
    patch was not, and that blur mismatch depresses NCC at the true location.  Matching the band
    and tightening the upper sigma 12 -> 8 takes the truth found among the 9 patches the pipeline
    gets wrong from 4/9 to 7/9, and the train score from 0.9513 to 0.9554."""
    f = np.asarray(f, np.float32); a = cv2.GaussianBlur(f, (0, 0), FB_LO) if FB_LO > 0 else f
    return a - cv2.GaussianBlur(a, (0, 0), FB_HI)
def bp_pre(f, hp=12):
    f = f.astype(np.float32); return f - cv2.GaussianBlur(f, (0, 0), hp, borderType=cv2.BORDER_REFLECT)
_yh, _xh = np.mgrid[:16, :16]
def raw_channel(small, p):
    """half-resolution brute force; returns (s1, x, y, relgap)"""
    q = fb_pre(p); MX = None; c = 15.5
    for sc in (0.92, 1.05, 1.2, 1.4):
        for a in range(0, 360, 10):
            th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th); u = (_xh - 7.5) * 2; v = (_yh - 7.5) * 2
            t = cv2.remap(q, (c + sc * (ca * u - sa * v)).astype(np.float32), (c + sc * (sa * u + ca * v)).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
            MX = r if MX is None else np.maximum(MX, r)
    pk = (MX == maximum_filter(MX, size=13)); ys, xs = np.nonzero(pk); o = np.argsort(-MX[ys, xs])[:2]
    s1, s2 = MX[ys[o[0]], xs[o[0]]], MX[ys[o[1]], xs[o[1]]]
    return float(s1), xs[o[0]] * 2 + 15, ys[o[0]] * 2 + 15, float((s1 - s2) / max(1 - s1, 1e-3))

# ----------------------------------------------------------------------------- scene matcher
class SceneMatcher:
    DSC = [0.85, 0.95, 1.05, 1.17, 1.3, 1.45]
    def __init__(self, im):
        self.im = im; self.imf = hp_pre(im); self.imraw = im.astype(np.float32)
        self.E = None
        if EXH_MODE != 'only':
            D = dense_desc(im); self.H, self.W, F = D.shape
            self.Df = D.reshape(-1, F); self.n2 = (self.Df * self.Df).sum(1)
        self.small = None; self.imbp = None
    def shortlist(self, p, n=3000):
        best = np.full(self.H * self.W, np.inf, np.float32)
        for sc in self.DSC:
            q = patch_desc(p, sc); np.minimum(best, self.n2 - 2 * self.Df @ q + (q * q).sum(), out=best)
        bm = -best.reshape(self.H, self.W); pk = (bm == maximum_filter(bm, size=5))
        ys, xs = np.nonzero(pk); o = np.argsort(-bm[ys, xs])[:n]
        return np.stack([xs[o], ys[o]], 1) * STRIDE
    def exh_peaks(self, p):
        key = hashlib.md5(self.im[::7, ::7].tobytes() + p.tobytes()).hexdigest()
        fn = os.path.join(CACHE, 'hp_' + key + '.npy')
        if os.path.exists(fn): return np.load(fn)
        if getattr(self, 'E', None) is None: self.E = Exhaustive(dog(self.im, 0.7, 3), scales=tuple(float(v) for v in os.environ.get('EXH_SCALES', '0.95,1.05').split(',')))
        best, arg, meta, off = self.E.search(dog(p, 0.7, 3)); pk = Exhaustive.peaks(best, off, 60)
        os.makedirs(CACHE, exist_ok=True); np.save(fn, pk); return pk
    def match(self, p, n_desc=300, n_s1=100, keep=10):
        pf = hp_pre(p)
        cands = []
        if EXH_MODE != 'off':
            pk = self.exh_peaks(p); self.last_exh = pk; cands.append(pk[:EXH_N, 1:3].astype(int))
        if EXH_MODE != 'only':
            sl = self.shortlist(p)
            C = match1(self.imf, templates1(pf), sl)
            cands += [sl[:n_desc], C[np.argsort(-C[:, 0])[:n_s1], 1:3].astype(int)]
        cand = np.vstack(cands)
        T, M, NMk, meta = templates3(pf)
        F = match3(self.imf, T, M, NMk, meta, cand); F = F[np.argsort(-F[:, 0])]
        out = []
        for k in range(len(F)):
            if all(np.hypot(F[k, 1] - F[j, 1], F[k, 2] - F[j, 2]) > 12 for j in out): out.append(k)
            if len(out) >= keep: break
        return F[out]
    def _bp(self):
        if self.small is None:
            self.imbp = fb_pre(self.imraw)                                     # full-res band-pass
            self.small = cv2.resize(self.imbp, (self.im.shape[1] // 2, self.im.shape[0] // 2),
                                    interpolation=cv2.INTER_AREA)
    def raw(self, p):
        self._bp(); return raw_channel(self.small, p)
    def raw_verified(self, p, npk=5, pad=6):
        """half-res band-pass search -> top peaks -> full-res masked-NCC verification."""
        self._bp()
        q = fb_pre(p); MX = None; c = 15.5
        for sc in (0.92, 1.05, 1.2, 1.4):
            for a in range(0, 360, 10):
                th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th); u = (_xh - 7.5) * 2; v = (_yh - 7.5) * 2
                t = cv2.remap(q, (c + sc * (ca * u - sa * v)).astype(np.float32),
                              (c + sc * (sa * u + ca * v)).astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                r = cv2.matchTemplate(self.small, t, cv2.TM_CCOEFF_NORMED)
                MX = r if MX is None else np.maximum(MX, r)
        pk = (MX == maximum_filter(MX, size=13)); ys, xs = np.nonzero(pk)
        o = np.argsort(-MX[ys, xs])[:npk]
        xy = np.stack([xs[o] * 2 + 15, ys[o] * 2 + 15], 1).astype(int)
        s1 = float(MX[ys[o[0]], xs[o[0]]]); s2 = float(MX[ys[o[1]], xs[o[1]]]) if len(o) > 1 else 0.0
        half = (s1, int(xy[0, 0]), int(xy[0, 1]), (s1 - s2) / max(1 - s1, 1e-3))
        T, M, NMk, meta = templates3(fb_pre(p))
        V = match3(self.imbp, T, M, NMk, meta, xy, pad=pad); V = V[np.argsort(-V[:, 0])]
        best = float(V[0, 0]); sec = float(V[1, 0]) if len(V) > 1 else 0.0
        return V, best, (best - sec) / max(1 - best, 1e-3), half

# ----------------------------------------------------------------------------- 2 evidence
_ya, _xa = np.mgrid[-45:46, -45:46]; _AN = (np.hypot(_xa, _ya) >= 18) & (np.hypot(_xa, _ya) <= 45)
def img_sim(imf, a, b):
    A = warp_crop(imf, a[1], a[2], 0, 1.0, sz=91)[_AN]; A = A - A.mean(); best = -1
    for da in (-6, -3, 0, 3, 6):
        B = warp_crop(imf, b[1], b[2], (b[3] - a[3]) + da, b[4] / a[4], sz=91)[_AN]; B = B - B.mean()
        best = max(best, float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-9)))
    return best
def dup_pair(imraw, F, thr=0.85, win=0.06):
    s = F[:, 0]; idx = [k for k in range(1, min(len(F), 6)) if s[k] >= s[0] - win]
    for k in idx:
        if img_sim(imraw, F[0], F[k]) >= thr: return (0, k)
    return None

def copy_move_pairs(im, hw=12, thr=0.97, min_contrast=25, maxn=6000, min_dist=40, vhw=30, vthr=0.95):
    f = im.astype(np.float32); g = cv2.GaussianBlur(f, (0, 0), 1.5); bg = cv2.GaussianBlur(f, (0, 0), 15)
    d = g - bg; pk = (d == maximum_filter(d, size=9)) & (d > min_contrast)
    ys, xs = np.nonzero(pk); H, W = f.shape
    ok = (xs >= vhw) & (xs < W - vhw) & (ys >= vhw) & (ys < H - vhw); xs, ys = xs[ok], ys[ok]
    o = np.argsort(-d[ys, xs])[:maxn]; xs, ys = xs[o], ys[o]
    V = np.array([f[y - hw:y + hw + 1, x - hw:x + hw + 1].ravel() for x, y in zip(xs, ys)])
    V = V - V.mean(1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-6
    out = []
    for c in range(0, len(V), 1000):
        S = V[c:c + 1000] @ V.T; ii, jj = np.nonzero(S > thr)
        for i, j in zip(ii + c, jj):
            if j <= i or np.hypot(xs[i] - xs[j], ys[i] - ys[j]) < min_dist: continue
            a = f[ys[i] - vhw:ys[i] + vhw + 1, xs[i] - vhw:xs[i] + vhw + 1]; best = (-1, 0, 0)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yj, xj = ys[j] + dy, xs[j] + dx
                    if yj - vhw < 0 or xj - vhw < 0 or yj + vhw + 1 > H or xj + vhw + 1 > W: continue
                    b = f[yj - vhw:yj + vhw + 1, xj - vhw:xj + vhw + 1]
                    a0 = a - a.mean(); b0 = b - b.mean(); v = (a0 * b0).sum() / np.sqrt((a0 * a0).sum() * (b0 * b0).sum() + 1e-6)
                    if v > best[0]: best = (v, dx, dy)
            if best[0] >= vthr: out.append((xs[i], ys[i], xs[j] + best[1], ys[j] + best[2]))
    return out

def star_catalog(im, n=300):
    f = im.astype(np.float32)
    bg = cv2.medianBlur(cv2.resize(im, (im.shape[1] // 8, im.shape[0] // 8), interpolation=cv2.INTER_AREA), 31)
    bg = cv2.resize(bg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    d = f - bg; g = cv2.GaussianBlur(d, (0, 0), 2.0); flux = cv2.blur(d, (13, 13)) * 169
    pk = (g == maximum_filter(g, size=15)) & (g > 20); ys, xs = np.nonzero(pk); o = np.argsort(-flux[ys, xs])[:n]
    return np.stack([xs[o], ys[o]], 1).astype(float)

RELGAP = 0.16
FB_S1_CAP = 0.9; FB_AGREE_RG = 0.16; FB_SOLO_RG = 0.30   # s1 cap applies to the solo rule only
FB_ADD_REL = 0.40; FB_REPLACE_REL = 0.4; CM_ANCHOR_MIN_DUP = 999   # always use pasted discs as anchors
def classify(F, dup):
    """-> kind, candidate indices"""
    s = F[:, 0]; rel = (s[0] - s[1]) / max(1 - s[0], 1e-3) if len(s) > 1 else 1.0
    if dup is not None:
        ks = sorted(set(dup) | {k for k in range(min(len(F), 6)) if s[k] >= s[0] - 0.01}); return 'dup', ks
    if rel >= RELGAP: return 'clear', [0]
    return 'flat', [k for k in range(min(len(F), OPT['flat_kmax'])) if s[k] >= s[0] - OPT['flat_delta']]

# ----------------------------------------------------------------------------- 3 identification
PAR = dict(cov_w=1.0, cov_p=0.30, cov_clip=-20.0, wd=10.0, wc=2.0, wf=1.0, cu=0.9, co=1.0, ws=(3.9, 3.7, 2.0), rb=(40, 120, 300), cs=3.0,
           sig_abs=10, sig_rel=0.007, startol=20, emin=700, emax=4500, img=3000, topk=150, cut=3.0, flat_top=0.6,
           pb_w=1.0, pb_c=2.97, pb_s=0.35, pb_clip=-20.0, unexpl_w=2.0)
def sim_fit(src, dst):
    ms, md = src.mean(), dst.mean(); s = src - ms; d = dst - md
    a = (np.conj(s) * d).sum() / max((np.abs(s) ** 2).sum(), 1e-9); return a, md - a * ms
def build_points(Ms, kinds_):
    Q = []; T = []; PID = []; ANC = []
    for i, (F, (kd, ks)) in enumerate(zip(Ms, kinds_)):
        if kd == 'flat' and F[0, 0] < PAR['flat_top']: continue
        use = ks if kd != 'flat' else ks[:3]
        for k in use:
            Q.append(F[k, 1] + 1j * F[k, 2]); T.append(kd); PID.append(i); ANC.append(kd != 'flat')
    return np.array(Q), np.array(T), np.array(PID), np.array(ANC, bool)
class _Ctx:
    def __init__(s, Q, T, PID, ANC, stars, p, npatch=0):
        s.npatch = int(npatch)
        o = np.argsort(PID, kind='stable'); s.Q, s.T, s.PID, s.ANC = Q[o], T[o], PID[o], ANC[o]
        s.WT = np.array([{'dup': p['wd'], 'clear': p['wc'], 'flat': p['wf']}[t] for t in s.T])
        # v6: copy-move anchors (PID >= npatch) are weaker evidence than a duplicated patch: on train only ~60% of
        # pasted-disc clusters sit on figure nodes, and validation scenes carry up to 24 clusters for <=27 nodes.
        if s.npatch: s.WT = np.where(PID[o] >= s.npatch, p.get('wcm', 4.0), s.WT)
        s.starts = np.r_[0, np.nonzero(np.diff(s.PID))[0] + 1]; s.p = p
        s.stars = np.asarray(stars[:p['rb'][-1]], float)
        s.sw = np.array([p['ws'][np.searchsorted(p['rb'], r, side='right')] for r in range(len(s.stars))])
        s.ST = cKDTree(s.stars) if len(s.stars) else None
        s.ndup = len(set(s.PID[(s.T == 'dup') & ((s.PID < s.npatch) | (s.npatch == 0))])) if len(s.T) else 0
def _pb(ctx, nin):
    """Patch-budget prior.  The issued patch count scales with the figure's in-frame node
    count: on train n_patches / in-frame nodes is 2.93, 2.83, 3.15 (sd 0.16).  Score a
    hypothesis by how far log(n_patches / nodes) sits from that centre.  pb_s is ~6x the
    observed spread so three scenes cannot impose a sharp prior, and the term is clipped
    like the duplicate-coverage one, so it settles near-ties and cannot overturn a
    confident fit.  Uses only the scene's own patch count, so it runs unchanged anywhere."""
    p = ctx.p
    if p['pb_w'] <= 0 or ctx.npatch <= 0: return 0.0
    z = (np.log(ctx.npatch / np.maximum(np.asarray(nin, float), 1.0)) - np.log(p['pb_c'])) / p['pb_s']
    return p['pb_w'] * np.maximum(-0.5 * z * z, p['pb_clip'])
def _unexpl(ctx, Mz, inside):
    """Penalty per duplicate-flagged patch that the hypothesis leaves unexplained.

    A `dup` patch is a near-certain figure star (0 false positives in 90 train cases), so a
    correct figure must put a node on one of its two copy locations.  The scorer already
    rewards nodes that match a patch; this is the converse, and it is the missing negative
    evidence.  Credit for the idea: a parallel implementation that measured it independently.
    Pure margin: on train the score is unchanged and the worst-case margin goes 18.0 -> 24.0."""
    w = ctx.p['unexpl_w']
    if w <= 0 or not len(ctx.T): return 0.0
    dup = np.unique(ctx.PID[(ctx.T == 'dup') & ((ctx.PID < ctx.npatch) | (ctx.npatch == 0))])
    if not len(dup): return 0.0
    cut = ctx.p['cut'] * np.maximum(ctx.p['sig_abs'], ctx.p['sig_rel'] * np.abs(ctx._a))
    un = np.zeros(Mz.shape[0])
    for g in dup:
        Qg = ctx.Q[ctx.PID == g]
        d = np.where(inside, np.abs(Mz[:, :, None] - Qg[None, None, :]).min(2), np.inf)
        un += (d.min(1) > cut).astype(float)
    return w * un
def _score(ctx, zp, a, b, ret=False):
    p = ctx.p; Q = ctx.Q
    Mz = a[:, None] * zp[None] + b[:, None]
    sig = np.maximum(p['sig_abs'], p['sig_rel'] * np.abs(a))[:, None]
    inside = (Mz.real > 0) & (Mz.real < p['img']) & (Mz.imag > 0) & (Mz.imag < p['img'])
    d2 = np.abs(Mz[:, :, None] - Q[None, None]) ** 2 / (2 * sig[:, :, None] ** 2)
    G = np.where(d2 < p['cut'] ** 2 / 2, ctx.WT[None, None] - d2, -np.inf)
    Gp = np.maximum.reduceat(G, ctx.starts, axis=2)
    gain = np.where(inside[:, :, None], Gp + p['cu'], -np.inf); gain = np.where(gain > 0, gain, 0)
    h, n, P = gain.shape; tot = np.zeros(h); g = gain.copy(); asg = -np.ones((h, n), int)
    for _ in range(min(n, P)):
        fl = g.reshape(h, -1); j = fl.argmax(1); v = fl[np.arange(h), j]
        if (v <= 0).all(): break
        nj, pj = np.divmod(j, P); ok = v > 0; tot += np.where(ok, v, 0)
        asg[np.arange(h)[ok], nj[ok]] = pj[ok]; g[np.arange(h), nj, :] = 0; g[np.arange(h), :, pj] = 0
    sc = tot - p['cu'] * inside.sum(1) - p['co'] * (~inside).sum(1)
    if p['cov_w'] > 0 and ctx.ndup > 0:
        # Duplicate-flagged patches are figure stars (0 false positives in 90 train cases), so the
        # figure must have enough in-frame nodes to produce them: D ~ Binomial(n_in, cov_p).
        # Bounded below, so this can only decide near-ties, never overturn a confident fit.
        from scipy.stats import binom
        lp = binom.logpmf(ctx.ndup, np.maximum(inside.sum(1), 1), p['cov_p'])
        sc = sc + p['cov_w'] * np.maximum(lp, p['cov_clip'])
    if ctx.ST is not None:
        pts = np.stack([Mz.real.ravel(), Mz.imag.ravel()], 1)
        tol = np.minimum(p['cut'] * sig, p['startol'] * np.maximum(1, np.abs(a) / 2000)[:, None])[:, 0]
        nb = ctx.ST.query_ball_point(pts, r=float(tol.max())); tolr = np.repeat(tol, Mz.shape[1])
        val = np.full(len(pts), -p['cs'])
        for i, lst in enumerate(nb):
            if lst:
                l = np.array(lst); d = np.hypot(ctx.stars[l, 0] - pts[i, 0], ctx.stars[l, 1] - pts[i, 1]); l = l[d < tolr[i]]
                if len(l): val[i] = ctx.sw[l].max()
        sc = sc + np.where(inside, val.reshape(Mz.shape), 0).sum(1)
    sc = sc + _pb(ctx, inside.sum(1))
    ctx._a = a; sc = sc - _unexpl(ctx, Mz, inside)
    return (sc, Mz, asg) if ret else sc
def _raster(Q, WT, stars, p, sig=12, pad=300, RS=4):
    img = p['img']; n = (img + 2 * pad) // RS; R = np.zeros((n, n), np.float32); rad = p['cut'] * sig
    yy, xx = np.mgrid[:n, :n]; cx = xx * RS - pad + RS / 2; cy = yy * RS - pad + RS / 2
    for q, w in zip(Q, WT):
        x0 = max(int((q.real + pad - rad) // RS), 0); x1 = min(int((q.real + pad + rad) // RS) + 1, n)
        y0 = max(int((q.imag + pad - rad) // RS), 0); y1 = min(int((q.imag + pad + rad) // RS) + 1, n)
        if x0 >= x1 or y0 >= y1: continue
        d2 = ((cx[y0:y1, x0:x1] - q.real) ** 2 + (cy[y0:y1, x0:x1] - q.imag) ** 2) / (2 * sig * sig)
        np.maximum(R[y0:y1, x0:x1], np.where(d2 < p['cut'] ** 2 / 2, w - d2 + p['cu'], 0), out=R[y0:y1, x0:x1])
    S = np.full((n, n), -p['cs'], np.float32); t = p['startol']
    for r, (x, y) in enumerate(stars[:p['rb'][-1]]):
        w = p['ws'][np.searchsorted(p['rb'], r, side='right')]
        x0 = max(int((x + pad - t) // RS), 0); x1 = min(int((x + pad + t) // RS) + 1, n)
        y0 = max(int((y + pad - t) // RS), 0); y1 = min(int((y + pad + t) // RS) + 1, n)
        m = ((cx[y0:y1, x0:x1] - x) ** 2 + (cy[y0:y1, x0:x1] - y) ** 2) < t * t
        sub = S[y0:y1, x0:x1]; sub[m] = np.maximum(sub[m], w)
    return R, S, pad, RS
def identify(P, Q, T, PID, ANC, stars, npatch=0, **kw):
    p = dict(PAR); p.update(kw); ctx = _Ctx(Q, T, PID, ANC, stars, p, npatch); Q, PID = ctx.Q, ctx.PID
    upid = np.unique(PID)
    R, S, pad, RS = _raster(Q, ctx.WT, ctx.stars, p); nR = R.shape[0]
    Aidx = np.nonzero(ctx.ANC)[0]
    pairs = np.array([(a, b) for a in Aidx for b in Aidx if PID[a] != PID[b]])
    res = []
    for name, N in P.items():
        zp0 = norm_pattern(N); n = len(zp0); I, J = np.triu_indices(n, 1); best = (-1e9, None)
        if len(pairs) == 0 or n < 2: res.append((best[0], name, None)); continue
        for refl in (False, True):
            zp = np.conj(zp0) if refl else zp0
            dz = (zp[J] - zp[I])[:, None]; dq = (Q[pairs[:, 1]] - Q[pairs[:, 0]])[None]
            al = dq / dz; be = Q[pairs[:, 0]][None] - al * zp[I][:, None]
            m = (np.abs(al) >= p['emin']) & (np.abs(al) <= p['emax']); al = al[m]; be = be[m]
            if len(al) == 0: continue
            Mz = al[:, None] * zp[None] + be[:, None]
            ix = np.clip(((Mz.real + pad) // RS).astype(int), 0, nR - 1); iy = np.clip(((Mz.imag + pad) // RS).astype(int), 0, nR - 1)
            ins = (Mz.real > 0) & (Mz.real < p['img']) & (Mz.imag > 0) & (Mz.imag < p['img'])
            rs = np.where(ins, R[iy, ix] + S[iy, ix], 0).sum(1) - p['cu'] * ins.sum(1) - p['co'] * (~ins).sum(1)
            if p['cov_w'] > 0 and ctx.ndup > 0:
                from scipy.stats import binom
                rs = rs + p['cov_w'] * np.maximum(binom.logpmf(ctx.ndup, np.maximum(ins.sum(1), 1), p['cov_p']), p['cov_clip'])
            rs = rs + _pb(ctx, ins.sum(1))
            k = min(p['topk'], len(rs)); top = np.argpartition(-rs, k - 1)[:k]
            a, b = al[top], be[top]; sc, Mz2, asg = _score(ctx, zp, a, b, True)
            a2 = a.copy(); b2 = b.copy()
            for h in range(len(a)):
                nodes = np.nonzero(asg[h] >= 0)[0]
                if len(nodes) < 3: continue
                tgt = []
                for j in nodes:
                    sel = np.nonzero(PID == upid[asg[h, j]])[0]; tgt.append(Q[sel][np.abs(Q[sel] - Mz2[h, j]).argmin()])
                a2[h], b2[h] = sim_fit(zp[nodes], np.array(tgt))
            ok = (np.abs(a2) >= p['emin']) & (np.abs(a2) <= p['emax'])
            sc2 = _score(ctx, zp, a2, b2); bt = ok & (sc2 > sc)
            sc = np.where(bt, sc2, sc); a = np.where(bt, a2, a); b = np.where(bt, b2, b)
            j = sc.argmax()
            if sc[j] > best[0]: best = (sc[j], (a[j], b[j], refl))
        res.append((best[0], name, best[1]))
    res.sort(key=lambda r: -r[0]); return res

# ----------------------------------------------------------------------------- 4 assignment
def decide(Ms, kinds_, P, ident_res, extra, relocate=None, tol_abs=30, tol_rel=0.02):
    n = len(Ms); pred = [None] * n
    name = ident_res[0][1]
    if ident_res[0][2] is None: return pred, name
    a, b, refl = ident_res[0][2]
    zp = norm_pattern(P[name]); zp = np.conj(zp) if refl else zp
    base = {'dup': 3.0, 'clear': 2.0, 'flat': 1.0}
    asg = []
    for it in range(3):
        M = a * zp + b; tol = max(tol_abs, tol_rel * abs(a)); G = np.full((n, len(M)), -1e6); pick = {}
        for i in range(n):
            kd, ks = kinds_[i]
            if kd == 'flat' and Ms[i][0, 0] < PAR['flat_top']: continue
            for k in ks:
                q = Ms[i][k, 1] + 1j * Ms[i][k, 2]; d = np.abs(M - q); g = base[kd] + Ms[i][k, 0] - 0.5 * (d / tol) ** 2
                for j in np.nonzero(d < tol)[0]:
                    if g[j] > G[i, j]: G[i, j] = g[j]; pick[(i, j)] = k
        r, c = linear_sum_assignment(-G)
        asg = [(i, j, pick[(i, j)]) for i, j in zip(r, c) if G[i, j] > -1e5]
        rel = [(i, j, k) for i, j, k in asg if kinds_[i][0] != 'flat']
        if len(rel) >= 3 and it < 2:
            a, b = sim_fit(zp[[j for i, j, k in rel]], np.array([Ms[i][k, 1] + 1j * Ms[i][k, 2] for i, j, k in rel]))
    for i, j, k in asg: pred[i] = (int(round(Ms[i][k, 1])), int(round(Ms[i][k, 2])), 1)
    relocate = relocate or {}
    for i in range(n):
        if pred[i] is None and kinds_[i][0] in ('dup', 'clear'):
            loc = relocate.get(i) or (Ms[i][0, 1], Ms[i][0, 2])
            pred[i] = (int(round(loc[0])), int(round(loc[1])), 0)
        elif pred[i] is None and extra.get(i) is not None:
            pred[i] = (int(extra[i][0]), int(extra[i][1]), 0)
    return pred, name


# ----------------------------------------------------------------------------- exhaustive full-res matcher


def dog(f, lo, hi):
    f = np.asarray(f, np.float32)
    a = cv2.GaussianBlur(f, (0, 0), lo, borderType=cv2.BORDER_REFLECT) if lo > 0 else f
    if hi is None or hi <= 0: return a
    return a - cv2.GaussianBlur(a, (0, 0), hi, borderType=cv2.BORDER_REFLECT)

class Exhaustive:
    """Full-resolution exhaustive masked-disk NCC over rotations x scales.
    numerator: plain correlation with zero-mean masked template; denominator: disk sums (once per radius)."""
    def __init__(self, img_pre, scales=(0.93, 1.0, 1.08), nang=36, Rmax=15):
        self.I = np.ascontiguousarray(img_pre, np.float32)
        self.scales = scales; self.nang = nang; self.Rmax = Rmax; self._den = {}
    def den(self, R):
        if R not in self._den:
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; M = ((xx * xx + yy * yy) <= R * R + 0.5).astype(np.float32)
            n = M.sum(); I = self.I
            S1 = cv2.matchTemplate(I, M, cv2.TM_CCORR); S2 = cv2.matchTemplate(I * I, M, cv2.TM_CCORR)
            sd = np.sqrt(np.maximum(S2 - S1 * S1 / n, 0)).astype(np.float32); fl = 0.5 * float(np.median(sd)) + 1e-3
            self._den[R] = (M, n, np.maximum(sd, fl))
        return self._den[R]
    def templates(self, p_pre, s, R, M, n):
        c = (p_pre.shape[0] - 1) / 2.; yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32); out = []
        for k in range(self.nang):
            a = 2 * np.pi * k / self.nang; ca, sa = np.cos(a), np.sin(a)
            mx = (c + s * (ca * xx - sa * yy)).astype(np.float32); my = (c + s * (sa * xx + ca * yy)).astype(np.float32)
            t = cv2.remap(p_pre, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            t = (t - (t * M).sum() / n) * M; t /= np.sqrt((t * t).sum()) + 1e-6
            out.append(t.astype(np.float32))
        return out
    def search(self, p_pre):
        best = None; arg = None; ti = 0; meta = []
        for s in self.scales:
            R = int(min(self.Rmax, np.floor(15.5 / s))); M, n, D = self.den(R); off = self.Rmax - R
            for k, t in enumerate(self.templates(p_pre, s, R, M, n)):
                r = cv2.matchTemplate(self.I, t, cv2.TM_CCORR) / D
                if off: r = r[off:r.shape[0] - off, off:r.shape[1] - off]
                if best is None: best = r; arg = np.zeros(r.shape, np.int16)
                else:
                    m = r > best; best[m] = r[m]; arg[m] = ti
                meta.append((360. * k / self.nang, s)); ti += 1
        return best, arg, np.array(meta), self.Rmax
    @staticmethod
    def peaks(best, off, n=20, nms=13):
        mx = maximum_filter(best, size=nms); ys, xs = np.nonzero(best == mx)
        v = best[ys, xs]; o = np.argsort(-v)[:n]
        return np.stack([v[o], xs[o] + off, ys[o] + off], 1)

# ----------------------------------------------------------------------------- PSF-adaptive star catalogue

def psf_hwhm(im, n=150):
    f = im.astype(np.float32); g = cv2.GaussianBlur(f, (0, 0), 1.5)
    bg = cv2.medianBlur(cv2.resize(im, (im.shape[1] // 8, im.shape[0] // 8), interpolation=cv2.INTER_AREA), 15)
    bg = cv2.resize(bg, (im.shape[1], im.shape[0])).astype(np.float32)
    d = g - bg; pk = (d == maximum_filter(d, 21)); pk[:30] = 0; pk[-30:] = 0; pk[:, :30] = 0; pk[:, -30:] = 0
    ys, xs = np.nonzero(pk); o = np.argsort(-d[ys, xs])[n//5:n]   # skip the most saturated
    yy, xx = np.mgrid[-25:26, -25:26]; rr = np.hypot(xx, yy); H = []
    for x, y in zip(xs[o], ys[o]):
        w = f[y - 25:y + 26, x - 25:x + 26]; b = np.median(w[rr > 20]); pkv = w[rr <= 1].mean() - b
        if pkv <= 5: continue
        prof = [w[(rr >= r - 0.5) & (rr < r + 0.5)].mean() - b for r in range(0, 20)]
        h = next((r for r, v in enumerate(prof) if v < pkv / 2), 20); H.append(h)
    return float(np.median(H)) if H else 3.0
def star_catalog2(im, n=300, hw=None, ret_all=False):
    f = im.astype(np.float32)
    if hw is None: hw = psf_hwhm(im)
    s = max(1.0, hw / 1.18)                          # gaussian sigma of PSF
    bg = cv2.medianBlur(cv2.resize(im, (im.shape[1] // 8, im.shape[0] // 8), interpolation=cv2.INTER_AREA), 15)
    bg = cv2.resize(bg, (im.shape[1], im.shape[0])).astype(np.float32)
    d = cv2.GaussianBlur(f - bg, (0, 0), s)
    R = int(np.ceil(max(4, 2.5 * s)))
    pk = (d == maximum_filter(d, 2 * R + 1)); m = 3 * R + 3
    pk[:m] = 0; pk[-m:] = 0; pk[:, :m] = 0; pk[:, -m:] = 0
    ys, xs = np.nonzero(pk); o = np.argsort(-d[ys, xs])[:8000]; ys, xs = ys[o], xs[o]
    W = 3 * R + 2; yy, xx = np.mgrid[-W:W + 1, -W:W + 1]; rr = np.hypot(xx, yy)
    ap = rr <= R; an = (rr >= 2 * R) & (rr <= 3 * R)
    win = np.stack([f[y - W:y + W + 1, x - W:x + W + 1] for x, y in zip(xs, ys)])
    b = np.median(win[:, an], axis=1); noise = 1.4826 * np.median(np.abs(win[:, an] - b[:, None]), axis=1) + 1
    flux = win[:, ap].sum(1) - b * ap.sum()
    snr = flux / (noise * np.sqrt(ap.sum()))
    score = flux * (snr > 5)
    o = np.argsort(-score)[:n]
    out = np.stack([xs[o], ys[o]], 1).astype(float)
    return (out, hw) if ret_all else out

# ----------------------------------------------------------------------------- per-patch evidence

_yy, _xx = np.mgrid[:32, :32] - 15.5; _rr = np.hypot(_xx, _yy)
ANN = (_rr >= 5) & (_rr <= 15)
def ann_score(imf, pf, x, y, a, s):
    best = -2; q = pf[ANN]; q = q - q.mean(); qn = np.sqrt((q * q).sum()) + 1e-9
    for da in (-4, -2, 0, 2, 4):
        for ds in (0.97, 1, 1.03):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    w = warp_crop(imf, x + dx, y + dy, a + da, s * ds)[ANN]; w = w - w.mean()
                    best = max(best, float((w * q).sum() / (np.sqrt((w * w).sum()) * qn + 1e-9)))
    return best
class Evidence:
    def __init__(self, im):
        self.im = im; self.SM = SceneMatcher(im); self.I_bp = dog(im, 1, 8); self.I_hp = self.SM.imf
    def patch(self, p, fallback=True):
        F = self.SM.match(p)                        # exhaustive candidates -> match3 -> 10 distinct
        pk = self.SM.last_exh
        f = dict(F=F, exh=pk[:30])
        x, y, a, s = F[0, 1], F[0, 2], F[0, 3], F[0, 4]
        f['ann_hp'] = ann_score(self.I_hp, hp_pre(p), x, y, -a, s)
        f['ann_bp'] = ann_score(self.I_bp, dog(p, 1, 8), x, y, -a, s)
        if len(F) > 1:
            x, y, a, s = F[1, 1], F[1, 2], F[1, 3], F[1, 4]
            f['ann_hp2'] = ann_score(self.I_hp, hp_pre(p), x, y, -a, s)
        if fallback: f['fb'] = self.SM.raw_verified(p)
        return f

def run_scene(d, cache):
    name = os.path.basename(os.path.normpath(d)); fn = os.path.join(cache, 'evid', name + '.pkl')
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    t0 = time.time(); im, ps = load_scene(d); PAR['img'] = int(max(im.shape))
    E = Evidence(im); pf = [E.patch(p) for p in ps]
    Ms = [f['F'] for f in pf]
    dups = [dup_pair(E.SM.imraw, F) for F in Ms]
    cm = copy_move_pairs(im)
    st1 = star_catalog(im, 1000); st2, hw = star_catalog2(im, 1000, ret_all=True)
    ev = dict(name=name, pf=pf, dups=dups, cm=cm, stars=st1, stars2=st2, hwhm=hw, n=len(ps), t=time.time() - t0)
    os.makedirs(os.path.dirname(fn), exist_ok=True); pickle.dump(ev, open(fn, 'wb'))
    return ev

# ----------------------------------------------------------------------------- decision

def cm_clusters(cm):
    cl = []
    for (x1, y1, x2, y2) in cm:
        v = np.array([x2 - x1, y2 - y1])
        if not any(np.abs(c[0] - v).max() <= 4 and np.hypot(c[1][0] - x1, c[1][1] - y1) < 80 for c in cl): cl.append((v, (x1, y1, x2, y2)))
    return cl
def star_weights(ev, stars, prior=(0.28, 0.44, 0.18, 0.10), alpha=6.0, tol=20.0, A=9e6, rb=(40, 120, 300)):
    """Per-scene likelihood-ratio star weights, calibrated on duplicate/copy-move locations (figure-star-like)."""
    pts = []
    for F, dd in zip([f['F'] for f in ev['pf']], ev['dups']):
        if dd is not None:
            for k in dd: pts.append((F[k, 1], F[k, 2]))
    c = np.zeros(4)
    for x, y in pts:
        d = np.hypot(stars[:, 0] - x, stars[:, 1] - y); w = np.nonzero(d < tol)[0]
        r = w[0] if len(w) else 10 ** 6; c[np.searchsorted(rb, r, side='right')] += 1
    p = (c + alpha * np.array(prior)) / (c.sum() + alpha)
    nb = np.diff(np.r_[0, rb]); q = np.minimum(nb * np.pi * tol * tol / A, 0.9); qn = 1 - q.sum()
    ws = tuple(np.log(p[:3] / q)); cs = -np.log(p[3] / qn)
    return ws, cs, c
def decide_ev(ev, relgap=None, fb=(None, None, None), ident_kw={}, use_cm=True, star_mode='v1', calib=False, flat_gate=None, force=None):
    g = globals(); old = (g['FB_ADD_REL'], g['FB_AGREE_RG'], g['FB_SOLO_RG'], g['RELGAP'])
    if fb[0] is not None: g['FB_ADD_REL'] = fb[0]
    if fb[1] is not None: g['FB_AGREE_RG'] = fb[1]
    if fb[2] is not None: g['FB_SOLO_RG'] = fb[2]
    if relgap is not None: g['RELGAP'] = relgap
    Ms = [f['F'].copy() for f in ev['pf']]; dups = list(ev['dups'])
    cm = ev['cm']
    if cm:
        A_ = np.array([(x1, y1) for x1, y1, x2, y2 in cm], float); B_ = np.array([(x2, y2) for x1, y1, x2, y2 in cm], float)
        for i, F in enumerate(Ms):
            if dups[i] is not None: continue
            s = F[:, 0]
            if s[0] < 0.75: continue
            for E1, E2 in ((A_, B_), (B_, A_)):
                dd = np.hypot(E1[:, 0] - F[0, 1], E1[:, 1] - F[0, 2]); j = dd.argmin()
                if dd[j] < 6:
                    row = np.array([[s[0], E2[j, 0], E2[j, 1], F[0, 3], F[0, 4]]], np.float32)
                    Ms[i] = np.vstack([F[:1], row, F[1:]]); dups[i] = (0, 1); break
    kinds_ = [classify(F, dd) for F, dd in zip(Ms, dups)]
    extra = {}; relocate = {}
    for i, (F, (kd, ks)) in enumerate(zip(Ms, kinds_)):
        if kd == 'dup' or 'fb' not in ev['pf'][i]: continue
        V, vbest, vrel, (s1, x, y, rg) = ev['pf'][i]['fb']; vloc = (float(V[0, 1]), float(V[0, 2]))
        if vrel >= FB_REPLACE_REL: relocate[i] = vloc
        if kd != 'flat' or not OPT['fb_extra']: continue
        d3 = np.hypot(F[:3, 1] - x, F[:3, 2] - y); k = int(d3.argmin())
        if vrel >= FB_ADD_REL: extra[i] = vloc
        elif d3[k] < 12 and rg >= FB_AGREE_RG: extra[i] = (float(F[k, 1]), float(F[k, 2]))
        elif rg >= FB_SOLO_RG and s1 < FB_S1_CAP: extra[i] = (float(x), float(y))
    Q, T, PID, ANC = build_points(Ms, kinds_)
    cl = cm_clusters(cm)
    if use_cm and cl:
        Q = list(Q); T = list(T); PID = list(PID); ANC = list(ANC); nid = len(Ms)   # copy-move anchors get PIDs >= npatch
        for _, (x1, y1, x2, y2) in cl:
            for (x, y) in ((x1, y1), (x2, y2)): Q.append(x + 1j * y); T.append('dup'); PID.append(nid); ANC.append(True)
            nid += 1
        Q = np.array(Q); T = np.array(T); PID = np.array(PID); ANC = np.array(ANC, bool)
    stars = ev['stars'] if star_mode == 'v1' else ev['stars2']
    kw = dict(ident_kw)
    if calib:
        ws, cs, cnt = star_weights(ev, stars); kw.update(ws=ws, cs=cs)
    PAR['img'] = 3000
    res = identify(P, Q, T, PID, ANC, stars, npatch=len(Ms), **kw)
    if force is not None: res = [r for r in res if r[1] == force] + [r for r in res if r[1] != force]
    pred, name = decide(Ms, kinds_, P, res, extra, relocate=relocate)
    g['FB_ADD_REL'], g['FB_AGREE_RG'], g['FB_SOLO_RG'], g['RELGAP'] = old
    return pred, name, res, kinds_
VARIANTS = [dict(star_mode='v1', calib=False), dict(star_mode='v1', calib=True),
            dict(star_mode='v2', calib=False), dict(star_mode='v2', calib=True)]
def decide_ens(ev, tau=4.0, variants=VARIANTS, verbose=False, **kw):
    outs = [decide_ev(ev, **v, **kw) for v in variants]
    names = sorted(P.keys()); prob = np.zeros(len(names))
    for pred, name, res, kinds in outs:
        sc = {r[1]: r[0] for r in res}; s = np.array([sc.get(n, -1e9) for n in names])
        e = np.exp((s - s.max()) / tau); prob += e / e.sum()
    prob /= len(outs); j = int(prob.argmax()); best = names[j]
    # use the variant that scores the chosen name highest for the assignment
    cand = [(dict((r[1], r) for r in o[2])[best][0], i) for i, o in enumerate(outs)]
    vi = max(cand)[1]; pred, name, res, kinds = outs[vi]
    if name != best:
        g = globals(); rr = dict((r[1], r) for r in res)[best]
        res2 = [rr] + [r for r in res if r[1] != best]
        pred, name, res, kinds = decide_ev(ev, **variants[vi], **kw, force=best)
    o = np.argsort(-prob)
    if verbose: print('   ens:', ' '.join(f'{names[k]}:{prob[k]:.2f}' for k in o[:4]))
    return pred, best, res, kinds, [(names[k], float(prob[k])) for k in o[:5]]


# ----------------------------------------------------------------------------- v5: multi-channel re-scoring
# Validation patches are ~2x noisier than train (patch/residual SNR ~2 vs ~3-4.5), and bright stars are
# saturated discs that a sigma=3 high-pass flattens.  Every candidate is therefore re-scored with the
# per-pose masked NCC on three channels - raw pixels, band-pass DoG(1,8) and the high-pass - and the
# three NCCs are averaged.  The pose search is restricted to the generator's measured scale range
# (0.93-1.13): confident matches in every scene, train and validation, sit at 0.93-1.08, while false
# matches pile up at the search-range limits (0.85 / 1.4) - letting them use those scales is what made
# look-alike stars competitive.
CHN = {'raw': (0, None), 'bp': (1, 8), 'hp': (0.7, 3), 'lp': (1.5, None)}
W_CH = {'raw': 1.0, 'bp': 1.0, 'hp': 1.0}
SC_R = np.array([0.93, 0.97, 1.0, 1.04, 1.08, 1.13])
RELGAP_V5 = 0.16
# probe switches (defaults = v7 exactly)
OPT = dict(relgap=0.16, fb_extra=True, flat_kmax=5, flat_delta=0.04, ens='all', patch_smooth=0.0)
def prep(a, k):
    lo, hi = CHN[k]; a = np.asarray(a, np.float32)
    if hi is None: return cv2.GaussianBlur(a, (0, 0), lo) if lo else a
    return dog(a, lo, hi)
_hu = (np.mgrid[:16, :16][1] - 7.5) * 2; _hv = (np.mgrid[:16, :16][0] - 7.5) * 2
def halfres_peaks(small, q, n=15):
    MX = None; c = 15.5
    for sc in (0.92, 1.05, 1.2):
        for a in range(0, 360, 10):
            th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th)
            t = cv2.remap(q, (c + sc * (ca * _hu - sa * _hv)).astype(np.float32), (c + sc * (sa * _hu + ca * _hv)).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED); MX = r if MX is None else np.maximum(MX, r)
    pk = (MX == maximum_filter(MX, size=13)); ys, xs = np.nonzero(pk); o = np.argsort(-MX[ys, xs])[:n]
    return np.stack([xs[o] * 2 + 15, ys[o] * 2 + 15], 1)
def templates3r(p, margin=0.5, scales=SC_R):
    c = (p.shape[0] - 1) / 2.; T = []; M = []; meta = []
    for s in scales:
        for a in ANG3:
            th = np.deg2rad(a); ca, sa = np.cos(th), np.sin(th)
            mx = (c + s * (ca * _x3 - sa * _y3)).astype(np.float32); my = (c + s * (sa * _x3 + ca * _y3)).astype(np.float32)
            valid = ((mx >= -margin) & (mx <= p.shape[1] - 1 + margin) & (my >= -margin) & (my <= p.shape[0] - 1 + margin)).ravel()[MI3].astype(np.float32)
            t = cv2.remap(p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[MI3]
            n = valid.sum(); t = (t - (t * valid).sum() / n) * valid; t /= np.linalg.norm(t) + 1e-6
            T.append(t); M.append(valid); meta.append((a, s))
    Mt = np.array(M, np.float32).T
    return np.array(T, np.float32).T, Mt, Mt.sum(0), np.array(meta)
def run_aug(d, ev, cache):
    name = os.path.basename(os.path.normpath(d)); fn = os.path.join(cache, 'aug', name + '.pkl')
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    im, ps = load_scene(d)
    IM = {k: prep(im, k) for k in CHN}
    SMALL = {k: cv2.resize(IM[k], (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA) for k in ('raw', 'lp')}
    out = []
    for i, p in enumerate(ps):
        f = ev['pf'][i]; F = f['F']; V = f['fb'][0]
        C = np.vstack([F[:, 1:3], V[:, 1:3]] + [halfres_peaks(SMALL[k], prep(p, k)) for k in ('raw', 'lp')]).astype(int)
        keep = []
        for c in C:
            if all(abs(c[0] - k[0]) > 3 or abs(c[1] - k[1]) > 3 for k in keep): keep.append(c)
        C = np.array(keep); S = {}
        for k in ('raw', 'bp', 'hp'):
            T, M, NMk, meta = templates3r(prep(p, k)); S[k] = match3(IM[k], T, M, NMk, meta, C, pad=3)
        out.append(dict(C=C, S=S))
    os.makedirs(os.path.dirname(fn), exist_ok=True); pickle.dump(out, open(fn, 'wb'))
    return out
def build_F(a, w=W_CH, pos='hp', keep=10):
    S = a['S']; comb = sum(w[k] * S[k][:, 0] for k in w) / sum(w.values()); P_ = S[pos]
    o = np.argsort(-comb); sel = []
    for j in o:
        if all(np.hypot(P_[j, 1] - P_[k, 1], P_[j, 2] - P_[k, 2]) > 12 for k in sel): sel.append(j)
        if len(sel) >= keep: break
    return np.stack([comb[sel], P_[sel, 1], P_[sel, 2], P_[sel, 3], P_[sel, 4]], 1).astype(np.float32)
def add_rawp(d, A, cache, sig):
    """probe: raw channel scored with a Gaussian-smoothed (denoised) patch, same candidates."""
    name = os.path.basename(os.path.normpath(d)); fn = os.path.join(cache, 'aug_rawp%.2f' % sig, name + '.pkl')
    if os.path.exists(fn): R = pickle.load(open(fn, 'rb'))
    else:
        im, ps = load_scene(d); I = im.astype(np.float32); R = []
        for a, p in zip(A, ps):
            T, M, NMk, meta = templates3r(cv2.GaussianBlur(p.astype(np.float32), (0, 0), sig)); R.append(match3(I, T, M, NMk, meta, a['C'], pad=3))
        os.makedirs(os.path.dirname(fn), exist_ok=True); pickle.dump(R, open(fn, 'wb'))
    return [dict(C=a['C'], S=dict(a['S'], raw=r)) for a, r in zip(A, R)]
def make_ev5(ev, A, im, w=W_CH, dup_thr=0.85):
    """Combined-channel candidate lists; duplicate evidence = union over the combined and high-pass rankings."""
    ev5 = dict(ev); ev5['pf'] = []; dups = []; imf = im.astype(np.float32)
    for f, a in zip(ev['pf'], A):
        F = build_F(a, w); g = dict(f); g['F'] = F; ev5['pf'].append(g); dups.append(dup_pair(imf, F, thr=dup_thr))
    for i, (f, a) in enumerate(zip(ev5['pf'], A)):
        if dups[i] is not None: continue
        Fb = build_F(a, {'hp': 1.0}); d = dup_pair(imf, Fb, thr=dup_thr)
        if d is not None:
            rows = [Fb[d[0]], Fb[d[1]]]; F = f['F']
            rest = [r for r in F if all(np.hypot(r[1] - q[1], r[2] - q[2]) > 12 for q in rows)]
            F2 = np.array(rows + rest[:8], np.float32); F2[:2, 0] = max(F[0, 0], Fb[d[0], 0])
            f['F'] = F2; dups[i] = (0, 1)
    ev5['dups'] = dups
    return ev5


def dedupe_pairs(ev5, rad=12.0):
    """v7: a sky star hosts at most one patch.  (1) Two duplicated patches on the same decoy pair: only the
    better-matching one keeps it (the other loses those rows, never gaining confidence); (2) copy-move pairs that
    coincide with a duplicated patch's pair are the same evidence and are not counted twice."""
    pf = ev5['pf']; dups = list(ev5['dups'])
    locs = {i: [(pf[i]['F'][k, 1], pf[i]['F'][k, 2]) for k in dups[i]] for i in range(len(pf)) if dups[i] is not None}
    order = sorted(locs, key=lambda i: -pf[i]['F'][0, 0]); kept = []
    for i in order:
        L = locs[i]
        clash = any(sum(min(np.hypot(x - u, y - v) for u, v in locs[j]) < rad for x, y in L) >= 2 for j in kept)
        if clash:
            F = pf[i]['F']; keep = np.array([min(np.hypot(r[1] - x, r[2] - y) for x, y in L) >= rad for r in F])
            if keep.sum() >= 2:
                F2 = F[keep].copy(); r0 = (F[0, 0] - F[1, 0]) / max(1 - F[0, 0], 1e-3)
                r1 = (F2[0, 0] - F2[1, 0]) / max(1 - F2[0, 0], 1e-3)
                if r1 > r0: F2[1, 0] = F2[0, 0] - r0 * (1 - F2[0, 0])
                pf[i] = dict(pf[i]); pf[i]['F'] = F2
            dups[i] = None
        else: kept.append(i)
    ev5['dups'] = dups
    D_ = [p for i in kept for p in locs[i]]
    if D_:
        D_ = np.array(D_)
        ev5['cm'] = [c for c in ev5['cm'] if not (np.hypot(D_[:, 0] - c[0], D_[:, 1] - c[1]).min() < rad or np.hypot(D_[:, 0] - c[2], D_[:, 1] - c[3]).min() < rad)]
    return ev5

# ----------------------------------------------------------------------------- scoring (local)

def _ramp(d): return float(np.clip((36 - d) / 24, 0, 1))
def _f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
def score_scene(truth, pred, tname, pname):
    tp = sum(t is not None and p is not None for t, p in zip(truth, pred)); fp = sum(t is None and p is not None for t, p in zip(truth, pred))
    fn = sum(t is not None and p is None for t, p in zip(truth, pred)); tn = len(truth) - tp - fp - fn
    pres = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
    L = [_ramp(np.hypot(p[0] - t[0], p[1] - t[1])) if p is not None else 0 for t, p in zip(truth, pred) if t is not None]
    loc = float(np.mean(L)) if L else 1.0
    figs = [t for t in truth if t is not None and t[2] == 1]; reps = [p for p in pred if p is not None]
    pairs = sorted((np.hypot(f[0] - r[0], f[1] - r[1]), i, j) for i, f in enumerate(figs) for j, r in enumerate(reps))
    uf, ur, tot = set(), set(), 0.0
    for d, i, j in pairs:
        if i in uf or j in ur: continue
        uf.add(i); ur.add(j); tot += _ramp(d)
    geo = tot / len(figs) if figs else 1.0
    ide = float(pname == tname)
    return dict(S=0.25 * pres + 0.2 * loc + 0.25 * geo + 0.3 * ide, presence=pres, localization=loc, geometric=geo, ident=ide)


# ----------------------------------------------------------------------------- main
def fmt(p): return '-1' if p is None else f'({p[0]}, {p[1]}, {p[2]})'
def scene_index(a):
    """Scenes to process, the output header and the number of patch columns.
    Uses an index CSV when one is available (this reproduces the competition format exactly); otherwise every
    sub-folder of <data>/<split>/ that contains <name>_image.png is a scene and its patches are counted on disk."""
    default = 'train_ground_truth.csv' if a.split == 'train' else 'sample_submission.csv'
    idx = a.index or os.path.join(a.data, default)
    if os.path.isfile(idx):
        rows = list(csv.DictReader(open(idx)))
        header = list(rows[0].keys())
        if 'constellation' not in header: header.append('constellation')
        ncols = len([h for h in header if h.startswith('patch_')])
        if ncols == 0:
            ncols = a.ncols or max(int(r['n_patches']) for r in rows)
            header = ['Id', 'n_patches'] + [f'patch_{k:02d}' for k in range(1, ncols + 1)] + ['constellation']
        print(f'scene list: {idx} ({len(rows)} scenes)', flush=True)
        return rows, header, ncols
    if a.index: raise FileNotFoundError(f'index CSV {a.index} not found')
    if a.eval: raise FileNotFoundError(f'--eval needs the ground-truth CSV {idx}')
    base = os.path.join(a.data, a.split)
    rows = []
    for d in sorted(glob.glob(os.path.join(base, '*', ''))):
        sid = os.path.basename(os.path.normpath(d))
        if os.path.isfile(os.path.join(d, f'{sid}_image.png')):
            rows.append({'Id': sid, 'n_patches': len(glob.glob(os.path.join(d, 'patches', 'patch_*.png')))})
    if not rows: raise FileNotFoundError(f'no scenes found: {idx} does not exist and no <scene>/<scene>_image.png under {base}/')
    ncols = a.ncols or max(r['n_patches'] for r in rows)
    header = ['Id', 'n_patches'] + [f'patch_{k:02d}' for k in range(1, ncols + 1)] + ['constellation']
    print(f'scene list: discovered {len(rows)} scene folders in {base}/ (no {default})', flush=True)
    return rows, header, ncols

def main():
    global P, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--split', default='validation')
    ap.add_argument('--out', default='submission.csv'); ap.add_argument('--eval', action='store_true')
    ap.add_argument('--cache', default='cache_final'); ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--relgap', type=float, default=0.16, help='confident-match threshold (v7: 0.16)')
    ap.add_argument('--no-fb-extra', action='store_true', help='drop second-channel presence calls for ambiguous patches')
    ap.add_argument('--flat-kmax', type=int, default=5); ap.add_argument('--flat-delta', type=float, default=0.04)
    ap.add_argument('--ens', default='all', choices=['all', 'calib'], help='identification ensemble variants')
    ap.add_argument('--patch-smooth', type=float, default=0.0, help='sigma of patch smoothing for the raw channel (0 = off)')
    ap.add_argument('--index', default=None, help='CSV listing the scenes (columns Id, n_patches[, patch_XX...]); default: '
                    'train_ground_truth.csv for --split train, sample_submission.csv otherwise; if that file does not exist '
                    'the scenes are discovered from the folders in <data>/<split>/')
    ap.add_argument('--ncols', type=int, default=None, help='number of patch_XX columns in the output when no index CSV '
                    'provides them (default: the largest patch count found)')
    a = ap.parse_args()
    OPT.update(relgap=a.relgap, fb_extra=not a.no_fb_extra, flat_kmax=a.flat_kmax, flat_delta=a.flat_delta, ens=a.ens, patch_smooth=a.patch_smooth)
    CACHE = os.path.join(a.cache, 'exh'); os.makedirs(CACHE, exist_ok=True)
    P = load_patterns(os.path.join(a.data, 'patterns'))
    rows, header, ncols = scene_index(a)
    if a.scenes: rows = [r for r in rows if r['Id'] in a.scenes]
    out = []; scores = []
    for r in rows:
        sid = r['Id']; t0 = time.time(); print(f'[{sid}]', flush=True)
        d = os.path.join(a.data, a.split, sid)
        ev = run_scene(d, a.cache); A = run_aug(d, ev, a.cache)
        if OPT['patch_smooth'] > 0: A = add_rawp(d, A, a.cache, OPT['patch_smooth'])
        ev5 = dedupe_pairs(make_ev5(ev, A, load_scene(d)[0]))
        PAR['img'] = 3000
        variants = VARIANTS if OPT['ens'] == 'all' else [v for v in VARIANTS if v['calib']]
        pred, name, res, kinds, top = decide_ens(ev5, relgap=OPT['relgap'], variants=variants)
        print('   -> %s  (%s)  present %d/%d  %.0fs' % (name, ', '.join(f'{n}:{p:.2f}' for n, p in top[:3]),
              sum(p is not None for p in pred), len(pred), time.time() - t0), flush=True)
        n = int(r['n_patches']); rec = {'Id': sid, 'n_patches': n}
        for k in range(1, ncols + 1): rec[f'patch_{k:02d}'] = fmt(pred[k - 1]) if k <= min(n, len(pred)) else '-1'
        rec['constellation'] = name; out.append(rec)
        if a.eval and a.split == 'train':
            truth = [None if r[f'patch_{k:02d}'].strip() == '-1' else ast.literal_eval(r[f'patch_{k:02d}']) for k in range(1, n + 1)]
            sc = score_scene(truth, pred[:n], r['constellation'], name); scores.append(sc['S'])
            print('   score', {k: round(v, 3) for k, v in sc.items()}, flush=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for rec in out: w.writerow(rec)
    print('wrote', a.out)
    if scores: print('MEAN SCORE %.4f over %d scenes' % (np.mean(scores), len(scores)))

if __name__ == '__main__':
    main()
