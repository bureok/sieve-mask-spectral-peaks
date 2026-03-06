from __future__ import annotations

import argparse
import math
import random
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]

EPS = 1e-12


def sieve_primes_upto(M: int) -> bytearray:
    """Return is_prime[n] for n in [0..M] as bytearray(0/1)."""
    is_prime = bytearray(b"\x01") * (M + 1)
    if M >= 0:
        is_prime[0] = 0
    if M >= 1:
        is_prime[1] = 0
    p = 2
    while p * p <= M:
        if is_prime[p]:
            start = p * p
            step = p
            is_prime[start : M + 1 : step] = b"\x00" * (((M - start) // step) + 1)
        p += 1
    return is_prime


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def build_candidates(M: int, W: int) -> array:
    """
    Candidate numbers n with gcd(n, W)=1, n>=2.
    NOTE: this excludes primes that divide W (e.g., 2,3,5 when W=30). That's intentional:
    we treat the wheel as the baseline "체" and model candidates conditional on gcd=1.
    """
    cands = array("I")
    for n in range(2, M + 1):
        if gcd(n, W) == 1:
            cands.append(n)
    return cands


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp_prob(p: float) -> float:
    if p < EPS:
        return EPS
    if p > 1.0 - EPS:
        return 1.0 - EPS
    return p


def fit_A_scale(cands_train: array, y_train: array) -> float:
    """
    Fit A in p = A / ln(n) on train by matching expected count to observed count.
    (1D monotonic solve via binary search.)
    """
    obs = int(sum(y_train))

    def expected(A: float) -> float:
        s = 0.0
        for n in cands_train:
            # We purposely avoid tiny-n special casing since wheel candidates are >= 2 and gcd=1.
            if n < 3:
                continue
            p = _clamp_prob(A / math.log(n))
            s += p
        return s

    lo, hi = 0.0, 1.0
    while expected(hi) < obs and hi < 1e6:
        hi *= 2.0

    for _ in range(70):
        mid = (lo + hi) / 2.0
        if expected(mid) < obs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_L(
    cands: array,
    y: array,
    A: float,
    W: int,
    alpha_by_residue: dict[int, float] | None,
) -> float:
    """Cross-entropy code length (nats)."""
    L = 0.0
    for n, yi in zip(cands, y):
        if n < 3:
            p = 1.0 - EPS if yi == 1 else EPS
        else:
            p0 = _clamp_prob(A / math.log(n))
            if alpha_by_residue is not None:
                a = alpha_by_residue.get(n % W, 0.0)
                p0 = _clamp_prob(sigmoid(logit(p0) + a))
            p = p0
        L -= yi * math.log(p) + (1 - yi) * math.log(1.0 - p)
    return L


def fit_alpha_by_residue(
    cands_train: array,
    y_train: array,
    A: float,
    W: int,
    iters: int = 25,
    lr: float = 1.0,
) -> dict[int, float]:
    """
    Fit alpha_r with a stable Newton update per residue.
    This is logistic regression with per-residue bias.
    """
    residues = sorted({int(n % W) for n in cands_train})
    ridx = {r: i for i, r in enumerate(residues)}
    alpha = [0.0] * len(residues)

    # Precompute per-sample baseline logit(p0) and residue index for speed.
    base_logit = array("d")
    r_index = array("I")
    y_vec = array("b")
    for n, yi in zip(cands_train, y_train):
        if n < 3:
            continue
        p0 = _clamp_prob(A / math.log(n))
        base_logit.append(logit(p0))
        r_index.append(ridx[int(n % W)])
        y_vec.append(int(yi))

    for _ in range(iters):
        g = [0.0] * len(residues)  # sum(y - p)
        h = [0.0] * len(residues)  # sum(p*(1-p))
        for bl, ri, yi in zip(base_logit, r_index, y_vec):
            i = int(ri)
            p = sigmoid(bl + alpha[i])
            g[i] += yi - p
            h[i] += p * (1.0 - p)
        for i in range(len(alpha)):
            denom = h[i] + 1e-9
            alpha[i] += lr * (g[i] / denom)

    return {r: alpha[i] for r, i in ridx.items()}


def block_permute(y: array, block_size: int, rng: random.Random) -> array:
    """Permute y by shuffling contiguous blocks."""
    blocks = [y[i : i + block_size] for i in range(0, len(y), block_size)]
    rng.shuffle(blocks)
    out = array("b")
    for b in blocks:
        out.extend(b)
    return out


def build_z_k(
    cands: array,
    y: array,
    A: float,
    W: int,
    alpha_by_residue: dict[int, float] | None,
) -> "np.ndarray":
    """
    z(k) = (b - p) / sqrt(p(1-p)) on candidate index axis (k-axis).
    p is baseline A/ln(n) optionally corrected by alpha(residue).
    """
    if np is None:
        raise RuntimeError("numpy is required for spectrum mode. Install with: python -m pip install numpy")

    z = np.empty(len(cands), dtype=np.float64)
    for i, (n, yi) in enumerate(zip(cands, y)):
        if n < 3:
            p = 1.0 - EPS if yi == 1 else EPS
        else:
            p = _clamp_prob(A / math.log(n))
            if alpha_by_residue is not None:
                a = alpha_by_residue.get(n % W, 0.0)
                p = _clamp_prob(sigmoid(logit(p) + a))
        denom = math.sqrt(max(EPS, p * (1.0 - p)))
        z[i] = (float(yi) - p) / denom
    return z


def make_window(window_name: str, L: int) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    if window_name == "hann":
        return np.hanning(L).astype(np.float64)
    if window_name == "blackman":
        return np.blackman(L).astype(np.float64)
    if window_name == "rect":
        return np.ones(L, dtype=np.float64)
    raise ValueError(f"unknown window: {window_name}")


def welch_psd(x: "np.ndarray", L: int, overlap: float = 0.5, window_name: str = "hann") -> "np.ndarray":
    """
    Welch PSD estimate (rfft bins).
    Returns mean PSD over segments (rfft bins).
    """
    if np is None:
        raise RuntimeError("numpy is required for spectrum mode.")
    if L <= 0:
        raise ValueError("L must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    if len(x) < L:
        raise ValueError("x shorter than window length L")

    step = int(L * (1.0 - overlap))
    step = max(1, step)
    win = make_window(window_name, L)
    win_norm = float(np.sum(win * win))
    win_norm = max(win_norm, EPS)

    acc = None
    count = 0
    for start in range(0, len(x) - L + 1, step):
        seg = x[start : start + L]
        seg = seg - float(np.mean(seg))  # remove DC
        segw = seg * win
        fft = np.fft.rfft(segw)
        psd = (np.abs(fft) ** 2) / win_norm
        acc = psd if acc is None else (acc + psd)
        count += 1

    if acc is None or count == 0:
        raise RuntimeError("Welch failed: no segments")
    return acc / float(count)


def pick_peaks(psd: "np.ndarray", K: int, f_min_ratio: float = 0.01) -> list[int]:
    """Pick top-K local maxima indices, skipping lowest f_min_ratio portion of bins."""
    n = int(psd.shape[0])
    if n < 3:
        return []
    i0 = int(n * f_min_ratio)
    i0 = max(1, min(i0, n - 2))

    cand: list[tuple[float, int]] = []
    for i in range(i0, n - 1):
        if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
            cand.append((float(psd[i]), i))
    cand.sort(reverse=True)
    return [idx for _, idx in cand[:K]]


def score_peaks(psd_test: "np.ndarray", peak_idxs: list[int], delta_bins: int = 3) -> float:
    """S = sum log(peak/base) where base is median(psd_test)."""
    base = float(np.median(psd_test)) if np is not None else 0.0
    base = max(base, 1e-30)
    n = int(psd_test.shape[0])
    s = 0.0
    for idx in peak_idxs:
        lo = max(0, idx - delta_bins)
        hi = min(n, idx + delta_bins + 1)
        peak = float(np.max(psd_test[lo:hi]))
        s += math.log(max(peak, 1e-300) / base)
    return s


def block_shuffle_np(x: "np.ndarray", block: int, rng: "np.random.Generator") -> "np.ndarray":
    n = int(x.shape[0])
    if block <= 0:
        raise ValueError("block must be positive")
    if block == 1:
        return rng.permutation(x)
    blocks = [x[i : i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return np.concatenate(blocks) if blocks else x.copy()


def cluster_peaks(peak_lists: list[list[int]], delta_bins: int) -> list[dict[str, object]]:
    """
    Greedy clustering of peak bin indices across windows.
    Returns clusters: {"members": list[(w, idx)], "center": int}
    """
    if np is None:
        raise RuntimeError("numpy is required for tracking mode.")
    clusters: list[dict[str, object]] = []
    for w, peaks in enumerate(peak_lists):
        for idx in peaks:
            placed = False
            for cl in clusters:
                center = int(cl["center"])  # type: ignore[assignment]
                if abs(int(idx) - center) <= delta_bins:
                    members = cl["members"]  # type: ignore[assignment]
                    assert isinstance(members, list)
                    members.append((w, int(idx)))
                    cl["center"] = int(np.median([m[1] for m in members]))
                    placed = True
                    break
            if not placed:
                clusters.append({"members": [(w, int(idx))], "center": int(idx)})
    return clusters


@dataclass(frozen=True)
class TrackingResult:
    N: int
    M: int
    W: int
    L: int
    overlap: float
    K: int
    f_min_ratio: float
    delta_bins: int
    num_windows: int
    window_len: int
    perms: int
    shuffle_block: int
    num_clusters: int
    persist_70: int
    persist_50: int
    mean_drift_bins: float
    score_real: float
    score_perm_mean: float
    score_perm_std: float
    score_perm_max: float
    p_value: float
    prom_score_real: float
    prom_score_perm_mean: float
    prom_score_perm_std: float
    prom_score_perm_max: float
    prom_p_value: float
    elapsed_s: float
    stable_peaks: list[dict[str, float | int]]


def peak_tracking_stats(
    z: "np.ndarray",
    *,
    num_windows: int,
    L: int,
    overlap: float,
    window_name: str = "hann",
    K: int,
    f_min_ratio: float,
    delta_bins: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """
    Split z into equal k-windows, compute Welch PSD per window, pick peaks, cluster peaks across windows.
    Returns (summary_stats, stable_peaks_list).
    stable_peaks_list: list of cluster summaries (center_bin, f, period, persistence, drift_std, hits)
    """
    if np is None:
        raise RuntimeError("numpy is required for tracking mode.")

    n = int(z.shape[0])
    if num_windows <= 1:
        raise ValueError("num_windows must be >= 2")
    window_len = n // num_windows
    if window_len < L:
        raise ValueError(f"window_len={window_len} < L={L}; reduce num_windows or reduce L")

    n_use = window_len * num_windows
    z_use = z[:n_use]

    peak_lists: list[list[int]] = []
    prom_by_window: list[dict[int, float]] = []
    for i in range(num_windows):
        seg = z_use[i * window_len : (i + 1) * window_len]
        psd = welch_psd(seg, L=L, overlap=overlap, window_name=window_name)
        base = float(np.median(psd))
        base = max(base, 1e-30)
        peaks = pick_peaks(psd, K=K, f_min_ratio=f_min_ratio)
        peak_lists.append(peaks)
        prom_map: dict[int, float] = {}
        for idx in peaks:
            prom_map[int(idx)] = float(psd[int(idx)] / base)
        prom_by_window.append(prom_map)

    clusters = cluster_peaks(peak_lists, delta_bins=delta_bins)

    pers: list[float] = []
    drifts: list[float] = []
    stable: list[dict[str, float | int]] = []
    prom_means: list[float] = []
    for cl in clusters:
        members = cl["members"]
        assert isinstance(members, list)
        windows = [w for w, _ in members]
        uniq_w = len(set(windows))
        persistence = uniq_w / float(num_windows)
        idxs = np.array([idx for _, idx in members], dtype=np.float64)
        drift = float(np.std(idxs)) if idxs.size else 0.0

        # prominence stats (peak height relative to median PSD in that window)
        proms: list[float] = []
        for w, idx in members:
            pm = prom_by_window[int(w)]
            proms.append(float(pm.get(int(idx), 1.0)))
        prom_mean = float(np.mean(proms)) if proms else 1.0
        prom_means.append(prom_mean)

        pers.append(persistence)
        drifts.append(drift)
        stable.append(
            {
                "center_bin": int(cl["center"]),
                "f": float(int(cl["center"]) / float(L)),
                "period": float(float(L) / float(int(cl["center"]))) if int(cl["center"]) != 0 else float("inf"),
                "persistence": float(persistence),
                "drift_std": float(drift),
                "hits": int(len(members)),
                "prom_mean": float(prom_mean),
            }
        )

    pers_arr = np.array(pers, dtype=np.float64) if pers else np.zeros(0, dtype=np.float64)
    drifts_arr = np.array(drifts, dtype=np.float64) if drifts else np.zeros(0, dtype=np.float64)

    persist_70 = int(np.sum(pers_arr >= 0.7)) if pers_arr.size else 0
    persist_50 = int(np.sum(pers_arr >= 0.5)) if pers_arr.size else 0
    mean_drift = float(np.mean(drifts_arr)) if drifts_arr.size else 0.0

    score = float(persist_70 * 2 + persist_50 - 0.1 * mean_drift)

    # prominence-based score: emphasize persistent peaks but weight by log-prominence
    prom_arr = np.array(prom_means, dtype=np.float64) if prom_means else np.zeros(0, dtype=np.float64)
    prom_arr = np.maximum(prom_arr, 1e-30)
    wts = (pers_arr >= 0.7).astype(np.float64) * 2.0 + (pers_arr >= 0.5).astype(np.float64) * 1.0
    prom_score = float(np.sum(wts * np.log(prom_arr))) - 0.1 * mean_drift

    stable.sort(key=lambda d: (-float(d["persistence"]), float(d["drift_std"]), -int(d["center_bin"])))
    summary: dict[str, float | int] = {
        "num_windows": int(num_windows),
        "window_len": int(window_len),
        "num_clusters": int(len(clusters)),
        "persist_70": int(persist_70),
        "persist_50": int(persist_50),
        "mean_drift_bins": float(mean_drift),
        "score": float(score),
        "prom_score": float(prom_score),
    }
    return summary, stable


def tracking_with_permutation(
    z_real: "np.ndarray",
    *,
    perms: int,
    shuffle_block: int,
    seed: int,
    num_windows: int,
    L: int,
    overlap: float,
    window_name: str = "hann",
    K: int,
    f_min_ratio: float,
    delta_bins: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]], float, float, float, float, float]:
    if np is None:
        raise RuntimeError("numpy is required for tracking mode.")
    rng = np.random.default_rng(seed)

    stats_real, stable = peak_tracking_stats(
        z_real,
        num_windows=num_windows,
        L=L,
        overlap=overlap,
        window_name=window_name,
        K=K,
        f_min_ratio=f_min_ratio,
        delta_bins=delta_bins,
    )
    score_real = float(stats_real["score"])
    prom_score_real = float(stats_real["prom_score"])

    scores = np.empty(perms, dtype=np.float64)
    prom_scores = np.empty(perms, dtype=np.float64)
    for i in range(perms):
        zp = block_shuffle_np(z_real, block=shuffle_block, rng=rng)
        st, _ = peak_tracking_stats(
            zp,
            num_windows=num_windows,
            L=L,
            overlap=overlap,
            window_name=window_name,
            K=K,
            f_min_ratio=f_min_ratio,
            delta_bins=delta_bins,
        )
        scores[i] = float(st["score"])
        prom_scores[i] = float(st["prom_score"])

    mean = float(np.mean(scores)) if perms else float("nan")
    std = float(np.std(scores, ddof=1)) if perms >= 2 else 0.0
    maxv = float(np.max(scores)) if perms else float("nan")
    count = int(np.sum(scores >= score_real))
    p_value = float((count + 1) / (perms + 1))

    pmean = float(np.mean(prom_scores)) if perms else float("nan")
    pstd = float(np.std(prom_scores, ddof=1)) if perms >= 2 else 0.0
    pmax = float(np.max(prom_scores)) if perms else float("nan")
    pcount = int(np.sum(prom_scores >= prom_score_real))
    pp_value = float((pcount + 1) / (perms + 1))

    # NOTE: We keep the return signature stable for existing callers by packing
    # both metrics into stats_real; callers that need prominence should read it from stats_real.
    stats_real["prom_score_real"] = float(prom_score_real)
    stats_real["prom_score_perm_mean"] = float(pmean)
    stats_real["prom_score_perm_std"] = float(pstd)
    stats_real["prom_score_perm_max"] = float(pmax)
    stats_real["prom_p_value"] = float(pp_value)

    return stats_real, stable, score_real, mean, std, maxv, p_value


def tracking_experiment(
    N: int,
    M: int,
    is_prime: bytearray,
    W: int,
    *,
    L: int = 65536,
    overlap: float = 0.5,
    K: int = 30,
    f_min_ratio: float = 0.01,
    delta_bins: int = 4,
    num_windows: int = 8,
    perms: int = 200,
    shuffle_block: int = 2048,
    seed: int = 0,
) -> TrackingResult:
    """
    3차 모델: peak frequency tracking (stability) + block-shuffle permutation.
    We compute z(k) over the *full* candidate sequence up to M (k-axis slicing).
    """
    if np is None:
        raise RuntimeError("numpy is required for tracking mode. Install with: python -m pip install numpy")

    t0 = time.time()

    cands_all = build_candidates(M, W)
    y_all = array("b")
    for n in cands_all:
        y_all.append(1 if is_prime[n] else 0)

    # Fit A on train portion (n <= N)
    c_train = array("I")
    y_train = array("b")
    for n, yi in zip(cands_all, y_all):
        if n <= N:
            c_train.append(n)
            y_train.append(yi)
        else:
            break

    A = fit_A_scale(c_train, y_train)
    z_all = build_z_k(cands_all, y_all, A, W, alpha_by_residue=None)

    stats_real, stable, score_real, mean, std, maxv, p_value = tracking_with_permutation(
        z_all,
        perms=perms,
        shuffle_block=shuffle_block,
        seed=seed,
        num_windows=num_windows,
        L=L,
        overlap=overlap,
        K=K,
        f_min_ratio=f_min_ratio,
        delta_bins=delta_bins,
    )

    elapsed = time.time() - t0
    stable_top = stable[:15]

    return TrackingResult(
        N=N,
        M=M,
        W=W,
        L=L,
        overlap=overlap,
        K=K,
        f_min_ratio=f_min_ratio,
        delta_bins=delta_bins,
        num_windows=int(stats_real["num_windows"]),
        window_len=int(stats_real["window_len"]),
        perms=perms,
        shuffle_block=shuffle_block,
        num_clusters=int(stats_real["num_clusters"]),
        persist_70=int(stats_real["persist_70"]),
        persist_50=int(stats_real["persist_50"]),
        mean_drift_bins=float(stats_real["mean_drift_bins"]),
        score_real=float(score_real),
        score_perm_mean=float(mean),
        score_perm_std=float(std),
        score_perm_max=float(maxv),
        p_value=float(p_value),
        prom_score_real=float(stats_real.get("prom_score_real", float("nan"))),
        prom_score_perm_mean=float(stats_real.get("prom_score_perm_mean", float("nan"))),
        prom_score_perm_std=float(stats_real.get("prom_score_perm_std", float("nan"))),
        prom_score_perm_max=float(stats_real.get("prom_score_perm_max", float("nan"))),
        prom_p_value=float(stats_real.get("prom_p_value", float("nan"))),
        elapsed_s=float(elapsed),
        stable_peaks=[{k: v for k, v in d.items()} for d in stable_top],
    )


@dataclass(frozen=True)
class SpectrumResult:
    N: int
    M: int
    W: int
    L: int
    K: int
    f_min_ratio: float
    delta_bins: int
    perms: int
    shuffle_block: int
    S_real: float
    S_perm_mean: float
    S_perm_std: float
    p_value: float
    train_len: int
    test_len: int
    elapsed_s: float


def spectrum_experiment(
    N: int,
    M: int,
    is_prime: bytearray,
    W: int,
    L: int = 65536,
    K: int = 30,
    f_min_ratio: float = 0.01,
    delta_bins: int = 3,
    perms: int = 200,
    shuffle_block: int = 2048,
    seed: int = 0,
) -> SpectrumResult:
    """
    2차 모델: residual z(k) 스펙트럼 재현성 테스트.
    - Build z(k) from null p(n)=A/ln(n) (no alpha by default).
    - Welch PSD on train to pick peak bins.
    - Score fixed peaks on test PSD.
    - Permutation on test by block-shuffling z(k).
    """
    if np is None:
        raise RuntimeError("numpy is required for spectrum mode. Install with: python -m pip install numpy")

    t0 = time.time()
    rng = np.random.default_rng(seed)

    cands_all = build_candidates(M, W)
    y_all = array("b")
    for n in cands_all:
        y_all.append(1 if is_prime[n] else 0)

    c_train = array("I")
    y_train = array("b")
    c_test = array("I")
    y_test = array("b")
    for n, yi in zip(cands_all, y_all):
        if n <= N:
            c_train.append(n)
            y_train.append(yi)
        else:
            c_test.append(n)
            y_test.append(yi)

    A = fit_A_scale(c_train, y_train)
    z_train = build_z_k(c_train, y_train, A, W, alpha_by_residue=None)
    z_test = build_z_k(c_test, y_test, A, W, alpha_by_residue=None)

    P_tr = welch_psd(z_train, L=L, overlap=0.5, window_name="hann")
    peak_idxs = pick_peaks(P_tr, K=K, f_min_ratio=f_min_ratio)

    P_te = welch_psd(z_test, L=L, overlap=0.5, window_name="hann")
    S_real = score_peaks(P_te, peak_idxs, delta_bins=delta_bins)

    S_perm: list[float] = []
    for _ in range(perms):
        zp = block_shuffle_np(z_test, block=shuffle_block, rng=rng)
        Pp = welch_psd(zp, L=L, overlap=0.5, window_name="hann")
        S_perm.append(score_peaks(Pp, peak_idxs, delta_bins=delta_bins))

    mean = float(np.mean(S_perm)) if S_perm else float("nan")
    std = float(np.std(S_perm, ddof=1)) if len(S_perm) >= 2 else 0.0
    count = sum(1 for s in S_perm if s >= S_real)
    p_value = (count + 1) / (perms + 1)

    elapsed = time.time() - t0
    return SpectrumResult(
        N=N,
        M=M,
        W=W,
        L=L,
        K=K,
        f_min_ratio=f_min_ratio,
        delta_bins=delta_bins,
        perms=perms,
        shuffle_block=shuffle_block,
        S_real=float(S_real),
        S_perm_mean=mean,
        S_perm_std=std,
        p_value=float(p_value),
        train_len=int(len(z_train)),
        test_len=int(len(z_test)),
        elapsed_s=float(elapsed),
    )


@dataclass(frozen=True)
class RunResult:
    N: int
    M: int
    W: int
    block_size: int
    perms: int
    iters: int
    lr: float
    test_candidates: int
    A: float
    delta_train_bits: float
    delta_nats: float
    delta_bits: float
    delta_bits_per_test_candidate: float
    p_value: float
    elapsed_s: float


def experiment(
    N: int,
    M: int,
    is_prime: bytearray,
    W: int,
    block_size: int,
    perms: int,
    iters: int,
    lr: float,
    seed: int,
) -> RunResult:
    t0 = time.time()
    rng = random.Random(seed)

    cands_all = build_candidates(M, W)

    y_all = array("b")
    for n in cands_all:
        y_all.append(1 if is_prime[n] else 0)

    # Split by numeric value, not by k.
    c_train = array("I")
    y_train = array("b")
    c_test = array("I")
    y_test = array("b")
    for n, yi in zip(cands_all, y_all):
        if n <= N:
            c_train.append(n)
            y_train.append(yi)
        else:
            c_test.append(n)
            y_test.append(yi)

    A = fit_A_scale(c_train, y_train)
    alpha = fit_alpha_by_residue(c_train, y_train, A, W, iters=iters, lr=lr)

    # Train improvement (sanity: should be >= 0 up to numerical noise)
    L0_train = compute_L(c_train, y_train, A, W, alpha_by_residue=None)
    L1_train = compute_L(c_train, y_train, A, W, alpha_by_residue=alpha)
    delta_train_bits = (L0_train - L1_train) / math.log(2)

    # Test improvement
    L0_test = compute_L(c_test, y_test, A, W, alpha_by_residue=None)
    L1_test = compute_L(c_test, y_test, A, W, alpha_by_residue=alpha)
    delta = L0_test - L1_test

    deltas_perm: list[float] = []
    for _ in range(perms):
        y_perm = block_permute(y_test, block_size, rng)
        L0p = compute_L(c_test, y_perm, A, W, alpha_by_residue=None)
        L1p = compute_L(c_test, y_perm, A, W, alpha_by_residue=alpha)
        deltas_perm.append(L0p - L1p)

    count = sum(1 for d in deltas_perm if d >= delta)
    p_value = (count + 1) / (perms + 1)

    elapsed = time.time() - t0
    delta_bits = delta / math.log(2)
    return RunResult(
        N=N,
        M=M,
        W=W,
        block_size=block_size,
        perms=perms,
        iters=iters,
        lr=lr,
        test_candidates=len(c_test),
        A=A,
        delta_train_bits=delta_train_bits,
        delta_nats=delta,
        delta_bits=delta_bits,
        delta_bits_per_test_candidate=delta_bits / max(1, len(c_test)),
        p_value=p_value,
        elapsed_s=elapsed,
    )


def render_md(results: list[RunResult], title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py` 실행 결과다.")
    lines.append("")

    if not results:
        lines.append("- (결과 없음)")
        lines.append("")
        return "\n".join(lines)

    # Group by W
    ws = sorted({r.W for r in results})
    r0 = results[0]

    lines.append("## 공통 설정")
    lines.append("")
    lines.append(f"- N: `{r0.N}`")
    lines.append(f"- M: `{r0.M}`")
    lines.append(f"- perms: `{r0.perms}`")
    lines.append(f"- iters: `{r0.iters}` / lr: `{r0.lr}`")
    lines.append(f"- W 목록: `{', '.join(map(str, ws))}`")
    lines.append("")

    lines.append("## 결과 요약")
    lines.append("")
    for W in ws:
        lines.append(f"### W = {W}")
        lines.append("")
        lines.append("| block_size | test_candidates | ΔL_train (bits) | ΔL_test (bits) | Δbits/test_cand | p-value | elapsed (s) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        subset = [r for r in results if r.W == W]
        subset.sort(key=lambda r: r.block_size)
        for r in subset:
            lines.append(
                f"| {r.block_size} | {r.test_candidates} | {r.delta_train_bits:.3f} | {r.delta_bits:.6f} | {r.delta_bits_per_test_candidate:.3e} | {r.p_value:.4g} | {r.elapsed_s:.2f} |"
            )
        lines.append("")

    lines.append("## 판정 가이드(관리자)")
    lines.append("")
    lines.append("- ΔL(bits) > 0 이고, block_size를 바꿔도 **비슷한 크기**로 유지되면 ‘잔차 구조’ 재현성에 유리.")
    lines.append("- p-value가 충분히 작아지면(예: < 0.05) 퍼뮤테이션 대비 의미있는 개선일 가능성이 상승.")
    lines.append("- block_size가 W의 배수면 퍼뮤테이션이 약해질 수 있으니(구조가 덜 깨질 수 있음) 피하는 걸 권장.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        type=str,
        default="alpha",
        choices=["alpha", "spectrum", "tracking", "validation", "sixth", "sieve_sweep", "mask_fft"],
    )
    p.add_argument("--N", type=int, default=2_000_000)
    p.add_argument("--M", type=int, default=3_000_000)
    p.add_argument("--W", type=int, default=30, help="Single wheel size (ignored if --Ws is set)")
    p.add_argument("--Ws", type=int, nargs="+", default=None, help="Wheel sizes sweep (e.g. 30 210)")
    p.add_argument("--W-list", type=int, nargs="+", default=None, dest="W_list", help="Alias of --Ws")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[997, 1009, 1201])
    p.add_argument("--perms", type=int, default=50)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="Newton step damping (1.0 = full step)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(ROOT / "riemann_experiment.md"))
    # Spectrum params
    p.add_argument("--L", type=int, default=65536)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--f-min-ratio", type=float, default=0.01)
    p.add_argument("--delta-bins", type=int, default=3)
    p.add_argument("--shuffle-block", type=int, default=2048)
    # Tracking params
    p.add_argument("--num-windows", type=int, default=8)
    p.add_argument("--track-delta-bins", type=int, default=4)
    # Validation params
    p.add_argument("--L-list", type=int, nargs="+", default=[32768, 65536, 131072])
    p.add_argument("--windows", type=str, nargs="+", default=["hann"])
    p.add_argument("--axes", type=str, nargs="+", default=["k"])
    # Sixth params
    p.add_argument("--max-lag", type=int, default=64)
    p.add_argument("--hi-band", type=float, nargs=2, default=[0.35, 0.5])
    p.add_argument("--mid-band", type=float, nargs=2, default=[0.1, 0.35])
    p.add_argument("--sixth-window", type=str, default="hann", choices=["hann", "blackman", "rect"])
    p.add_argument("--sixth-L", type=int, default=65536)
    # Sieve sweep params (cause tracing)
    p.add_argument("--sieve-primes", type=int, nargs="+", default=[7, 11, 13, 17, 19, 23, 29])
    p.add_argument("--sieve-steps", type=int, default=0, help="0 = use all primes in --sieve-primes")
    p.add_argument("--sieve-tol", type=float, default=5e-4)
    p.add_argument("--sieve-topk", type=int, default=12)
    # Mask FFT params (9차: 왜 하필 그 h 인가?)
    p.add_argument("--mask-max-pn", type=int, default=300000, help="Skip steps with Pn larger than this")
    p.add_argument("--mask-topk", type=int, default=25, help="Top-K DFT bins to show per step")
    p.add_argument("--mask-ablate", action="store_true", help="For each step, drop one rune(prime) and compare target energy")
    p.add_argument(
        "--mask-swap-last",
        action="store_true",
        help="For each step, swap the last included rune with the next unused rune and compare",
    )
    p.add_argument(
        "--mask-blur-block",
        type=int,
        default=0,
        help="If >0, blur boundary by shuffling within blocks of this size (keeps 0/1 counts)",
    )
    p.add_argument(
        "--mask-blur-sweep",
        type=int,
        nargs="+",
        default=None,
        help="If set, run blur_block sweep and summarize (e.g. 64 128 256 512 1024)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.N >= args.M:
        raise SystemExit("N must be < M")

    Ws_in = args.W_list if args.W_list else args.Ws
    Ws = Ws_in if Ws_in else [args.W]
    Ws = [int(w) for w in Ws]

    is_prime = sieve_primes_upto(args.M)

    out_path = Path(args.out)

    if args.mode == "alpha":
        results: list[RunResult] = []
        for W in Ws:
            for bs in args.block_sizes:
                res = experiment(
                    N=args.N,
                    M=args.M,
                    is_prime=is_prime,
                    W=W,
                    block_size=bs,
                    perms=args.perms,
                    iters=args.iters,
                    lr=args.lr,
                    seed=args.seed + (W * 1000003) + bs,
                )
                results.append(res)

        out_path.write_text(
            render_md(
                results,
                title=f"Riemann-ish compression experiment (N={args.N}, M={args.M})",
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote: {out_path}")
        return

    if args.mode == "spectrum":
        if np is None:
            raise SystemExit("Spectrum mode requires numpy. Install with: python -m pip install numpy")

        lines: list[str] = []
        lines.append(f"# Riemann-ish spectrum experiment (N={args.N}, M={args.M})")
        lines.append("")
        lines.append("이 문서는 `tools/riemann_experiment.py --mode spectrum` 실행 결과다.")
        lines.append("")
        lines.append("## 설정")
        lines.append("")
        lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
        lines.append(f"- L: `{args.L}` / overlap: `0.5` / window: `Hann`")
        lines.append(
            f"- K(peaks): `{args.K}` / f_min_ratio: `{args.f_min_ratio}` / delta_bins: `{args.delta_bins}`"
        )
        lines.append(f"- perms: `{args.perms}` / shuffle_block: `{args.shuffle_block}`")
        lines.append("")

        lines.append("## 결과")
        lines.append("")
        lines.append("| W | train_len | test_len | S_real | S_perm_mean | S_perm_std | p-value | elapsed (s) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for W in Ws:
            r = spectrum_experiment(
                N=args.N,
                M=args.M,
                is_prime=is_prime,
                W=W,
                L=args.L,
                K=args.K,
                f_min_ratio=args.f_min_ratio,
                delta_bins=args.delta_bins,
                perms=args.perms,
                shuffle_block=args.shuffle_block,
                seed=args.seed + (W * 1000003) + 17,
            )
            lines.append(
                f"| {r.W} | {r.train_len} | {r.test_len} | {r.S_real:.3f} | {r.S_perm_mean:.3f} | {r.S_perm_std:.3f} | {r.p_value:.4g} | {r.elapsed_s:.2f} |"
            )

        lines.append("")
        lines.append("## 판정 가이드(2차 모델)")
        lines.append("")
        lines.append("- 학습에서 고른 피크가 검증에서 살아있으면 S_real가 perm 분포 상단 꼬리로 튄다(p-value ↓).")
        lines.append("- p-value가 W=30,210에서 **둘 다** 낮게 나오면 재현성 신호로 볼 근거가 강해진다.")
        lines.append("")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")
        return

    if args.mode == "tracking":
        _run_tracking(args, Ws, is_prime, out_path)
        return

    if args.mode == "validation":
        _run_validation(args, Ws, is_prime, out_path)
        return

    if args.mode == "sixth":
        _run_sixth(args, Ws, is_prime, out_path)
        return

    if args.mode == "sieve_sweep":
        _run_sieve_sweep(args, Ws, is_prime, out_path)
        return

    if args.mode == "mask_fft":
        _run_mask_fft(args, Ws, out_path)
        return

    raise SystemExit(f"Unknown mode: {args.mode}")


def _standardize_np(x: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return x - mu
    return (x - mu) / sd


def _acf_lags(x: "np.ndarray", max_lag: int) -> dict[int, float]:
    """
    Fast ACF estimate for small lags using dot products.
    x should be standardized (mean 0, var 1).
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    out: dict[int, float] = {}
    for lag in range(1, int(max_lag) + 1):
        if lag >= n:
            break
        out[lag] = float(np.mean(x[:-lag] * x[lag:]))
    return out


def _hi_mid_ratio_H_from_psd(
    psd: "np.ndarray",
    *,
    L: int,
    hi_band: tuple[float, float],
    mid_band: tuple[float, float],
) -> float:
    """
    H = log(E_hi / E_mid) where energies are sums of PSD bins.
    Frequency normalization uses f = bin/L.
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    nbin = int(psd.shape[0])
    # valid bins: 0..L/2 -> frequencies 0..0.5
    def idx_range(f0: float, f1: float) -> tuple[int, int]:
        lo = int(math.ceil(max(0.0, f0) * L))
        hi = int(math.floor(min(0.5, f1) * L))
        lo = max(1, min(lo, nbin - 1))
        hi = max(lo, min(hi, nbin - 1))
        return lo, hi

    hi0, hi1 = idx_range(*hi_band)
    mi0, mi1 = idx_range(*mid_band)
    E_hi = float(np.sum(psd[hi0 : hi1 + 1]))
    E_mid = float(np.sum(psd[mi0 : mi1 + 1]))
    return float(math.log(max(E_hi, EPS) / max(E_mid, EPS)))


def _hi_peak_score_from_psd(
    psd: "np.ndarray",
    *,
    L: int,
    band: tuple[float, float],
    K: int = 20,
) -> float:
    """
    Scalar capturing *narrow* peaks in a band:
    S_hi = sum_{top-K local maxima in band} log( psd[idx] / median(psd) ).
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    base = float(np.median(psd))
    base = max(base, 1e-30)
    nbin = int(psd.shape[0])

    lo = int(math.ceil(max(0.0, band[0]) * L))
    hi = int(math.floor(min(0.5, band[1]) * L))
    lo = max(1, min(lo, nbin - 2))
    hi = max(lo, min(hi, nbin - 2))

    cand: list[tuple[float, int]] = []
    for i in range(lo, hi + 1):
        if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
            cand.append((float(psd[i]), i))
    cand.sort(reverse=True)
    top = cand[: int(K)]

    s = 0.0
    for val, _i in top:
        s += math.log(max(val / base, 1e-30))
    return float(s)


def _H_stat(
    x: "np.ndarray",
    *,
    L: int,
    window_name: str,
    hi_band: tuple[float, float],
    mid_band: tuple[float, float],
) -> float:
    if np is None:
        raise RuntimeError("numpy is required.")
    xz = _standardize_np(x)
    psd = welch_psd(xz, L=int(L), overlap=0.5, window_name=window_name)
    return _hi_mid_ratio_H_from_psd(psd, L=int(L), hi_band=hi_band, mid_band=mid_band)


def _S_hi_stat(
    x: "np.ndarray",
    *,
    L: int,
    window_name: str,
    band: tuple[float, float],
    K: int = 20,
) -> float:
    if np is None:
        raise RuntimeError("numpy is required.")
    xz = _standardize_np(x)
    psd = welch_psd(xz, L=int(L), overlap=0.5, window_name=window_name)
    return _hi_peak_score_from_psd(psd, L=int(L), band=band, K=int(K))


def _H_perm_pvalue(
    x: "np.ndarray",
    *,
    perms: int,
    shuffle_block: int,
    seed: int,
    L: int,
    window_name: str,
    hi_band: tuple[float, float],
    mid_band: tuple[float, float],
) -> tuple[float, float]:
    """
    Return (H_real, p_value) where perm distribution is generated by block-shuffling x.
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    rng = np.random.default_rng(int(seed))
    H_real = _H_stat(x, L=L, window_name=window_name, hi_band=hi_band, mid_band=mid_band)
    if perms <= 0:
        return H_real, float("nan")
    count = 0
    for _ in range(int(perms)):
        xp = block_shuffle_np(np.asarray(x, dtype=np.float64), block=int(shuffle_block), rng=rng)
        Hp = _H_stat(xp, L=L, window_name=window_name, hi_band=hi_band, mid_band=mid_band)
        if Hp >= H_real:
            count += 1
    p = float((count + 1) / (int(perms) + 1))
    return H_real, p


def _S_hi_perm_pvalue(
    x: "np.ndarray",
    *,
    perms: int,
    shuffle_block: int,
    seed: int,
    L: int,
    window_name: str,
    band: tuple[float, float],
    K: int = 20,
) -> tuple[float, float]:
    """
    Return (S_hi_real, p_value) where perm distribution is generated by block-shuffling x.
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    rng = np.random.default_rng(int(seed))
    S_real = _S_hi_stat(x, L=L, window_name=window_name, band=band, K=K)
    if perms <= 0:
        return S_real, float("nan")
    count = 0
    for _ in range(int(perms)):
        xp = block_shuffle_np(np.asarray(x, dtype=np.float64), block=int(shuffle_block), rng=rng)
        Sp = _S_hi_stat(xp, L=L, window_name=window_name, band=band, K=K)
        if Sp >= S_real:
            count += 1
    p = float((count + 1) / (int(perms) + 1))
    return S_real, p


def _run_sixth(args: argparse.Namespace, Ws: list[int], is_prime: bytearray, out_path: Path) -> None:
    """
    6차: ACF(짧은 래그) + hi-band ratio(H) + (가능하면) b_stage vs z_stage 원인 분리.
    Datasets: real / bernoulli_null / shuffle_null / phase_null.
    Stages: b_stage (raw 0/1), z_stage (residual standardized via p).
    """
    if np is None:
        raise SystemExit("Sixth mode requires numpy. Install with: python -m pip install numpy")

    hi_band = (float(args.hi_band[0]), float(args.hi_band[1]))
    mid_band = (float(args.mid_band[0]), float(args.mid_band[1]))

    lines: list[str] = []
    lines.append(f"# 6th experiment: ACF + hi-band ratio (N={args.N}, M={args.M})")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py --mode sixth` 실행 결과다.")
    lines.append("")
    lines.append("## 설정")
    lines.append("")
    lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
    lines.append(f"- ACF max_lag: `{args.max_lag}`")
    lines.append(f"- H bands: hi=`[{hi_band[0]}, {hi_band[1]}]`, mid=`[{mid_band[0]}, {mid_band[1]}]`")
    lines.append(f"- H uses Welch: L=`{args.sixth_L}`, window=`{args.sixth_window}`, overlap=`0.5`")
    lines.append(f"- perms(H): `{args.perms}` / shuffle_block: `{args.shuffle_block}` / seed: `{args.seed}`")
    lines.append("")

    lags_focus = [2, 3, 4, 5, 6, 7, 8]
    acf_rows: list[str] = []
    h_rows: list[str] = []
    trk_rows: list[str] = []

    for W in Ws:
        cands_all = build_candidates(args.M, W)
        y_all = array("b")
        for n in cands_all:
            y_all.append(1 if is_prime[n] else 0)

        # Fit A on train (n<=N)
        c_train = array("I")
        y_train = array("b")
        for n, yi in zip(cands_all, y_all):
            if n <= args.N:
                c_train.append(n)
                y_train.append(yi)
            else:
                break
        A = fit_A_scale(c_train, y_train)

        # Build k-axis arrays
        b_real = np.frombuffer(y_all, dtype=np.int8).astype(np.float64, copy=True)
        p = _build_p_for_candidates(cands_all, A)
        r_real = b_real - p
        z_real = _build_z_from_b_p(b_real, p)

        rng = np.random.default_rng(int(args.seed + (W * 1000003) + 123))
        b_bern = rng.binomial(1, np.clip(p, EPS, 1.0 - EPS)).astype(np.float64)
        r_bern = b_bern - p
        z_bern = _build_z_from_b_p(b_bern, p)
        b_shuffle = block_shuffle_np(b_real, block=int(args.shuffle_block), rng=rng)
        r_shuffle = block_shuffle_np(r_real, block=int(args.shuffle_block), rng=rng)
        z_shuffle = block_shuffle_np(z_real, block=int(args.shuffle_block), rng=rng)
        # phase surrogate on z; for b_stage we phase-randomize standardized b (float surrogate)
        z_phase = phase_randomize_real(z_real, seed=int(args.seed + (W * 1000003) + 991))
        b_phase = phase_randomize_real(_standardize_np(b_real), seed=int(args.seed + (W * 1000003) + 992))
        r_phase = phase_randomize_real(_standardize_np(r_real), seed=int(args.seed + (W * 1000003) + 993))

        datasets_stage: dict[str, dict[str, "np.ndarray"]] = {
            "real": {"b_stage": b_real, "r_stage": r_real, "z_stage": z_real},
            "bernoulli_null": {"b_stage": b_bern, "r_stage": r_bern, "z_stage": z_bern},
            "shuffle_null": {"b_stage": b_shuffle, "r_stage": r_shuffle, "z_stage": z_shuffle},
            "phase_null": {"b_stage": b_phase, "r_stage": r_phase, "z_stage": z_phase},
        }

        for dataset_name, stages in datasets_stage.items():
            for stage_name, x in stages.items():
                x_std = _standardize_np(x)
                acf = _acf_lags(x_std, int(args.max_lag))
                row = [f"{acf.get(l, float('nan')):.6f}" for l in lags_focus]
                acf_rows.append(f"| {dataset_name} | {W} | {stage_name} | " + " | ".join(row) + " |")

                H, pH = _H_perm_pvalue(
                    x,
                    perms=int(args.perms),
                    shuffle_block=int(args.shuffle_block),
                    seed=int(
                        args.seed
                        + (W * 1000003)
                        + (
                            {
                                "real": 1,
                                "bernoulli_null": 2,
                                "shuffle_null": 3,
                                "phase_null": 4,
                            }[dataset_name]
                            * 10007
                        )
                        + (1 if stage_name == "b_stage" else (2 if stage_name == "r_stage" else 3)) * 37
                    ),
                    L=int(args.sixth_L),
                    window_name=str(args.sixth_window),
                    hi_band=hi_band,
                    mid_band=mid_band,
                )
                S, pS = _S_hi_perm_pvalue(
                    x,
                    perms=int(args.perms),
                    shuffle_block=int(args.shuffle_block),
                    seed=int(
                        args.seed
                        + (W * 1000003)
                        + (
                            {
                                "real": 1,
                                "bernoulli_null": 2,
                                "shuffle_null": 3,
                                "phase_null": 4,
                            }[dataset_name]
                            * 10007
                        )
                        + (1 if stage_name == "b_stage" else (2 if stage_name == "r_stage" else 3)) * 97
                    ),
                    L=int(args.sixth_L),
                    window_name=str(args.sixth_window),
                    band=hi_band,
                    K=20,
                )
                pH_str = f"{pH:.4g}" if math.isfinite(pH) else "NA"
                pS_str = f"{pS:.4g}" if math.isfinite(pS) else "NA"
                h_rows.append(f"| {dataset_name} | {W} | {stage_name} | {H:.6f} | {pH_str} | {S:.3f} | {pS_str} |")

        # tracking on z_stage only (uses tracking_with_permutation which always permutes by shuffle)
        for dataset_name in ["real", "phase_null", "shuffle_null", "bernoulli_null"]:
            z = datasets_stage[dataset_name]["z_stage"]
            stats_real, _stable, score_real, _mean, _std, _maxv, p_val = tracking_with_permutation(
                np.asarray(z, dtype=np.float64),
                perms=int(min(200, args.perms)),  # keep bounded for runtime
                shuffle_block=int(args.shuffle_block),
                seed=int(args.seed + (W * 1000003) + ({"real": 1, "phase_null": 2, "shuffle_null": 3, "bernoulli_null": 4}[dataset_name] * 999)),
                num_windows=int(args.num_windows),
                L=int(args.sixth_L),
                overlap=0.5,
                window_name=str(args.sixth_window),
                K=int(args.K),
                f_min_ratio=float(args.f_min_ratio),
                delta_bins=int(args.track_delta_bins),
            )
            prom_score = float(stats_real.get("prom_score_real", float("nan")))
            prom_p = float(stats_real.get("prom_p_value", float("nan")))
            trk_rows.append(f"| {dataset_name} | {W} | {score_real:.3f} | {prom_score:.3f} | {p_val:.4g} | {prom_p:.4g} |")

    # Render sections
    lines.append("## 6.1 ACF 요약 (lag 2..8)")
    lines.append("")
    lines.append("| dataset | W | stage | " + " | ".join([f"lag{l}" for l in lags_focus]) + " |")
    lines.append("|---|---:|---|" + "|".join(["---:"] * len(lags_focus)) + "|")
    lines.extend(acf_rows)
    lines.append("")

    lines.append("## 6.2 Hi-band ratio (H=log(Ehi/Emid)) + hi-band peak score (S_hi)")
    lines.append("")
    lines.append("| dataset | W | stage | H | p(H) | S_hi | p(S_hi) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    lines.extend(h_rows)
    lines.append("")

    lines.append("## 6.3 tracking (z_stage only, 참고)")
    lines.append("")
    lines.append("| dataset | W | score | prom_score | p | prom_p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.extend(trk_rows)

    lines.append("")
    lines.append("## 6.4 자동 판정(가이드)")
    lines.append("")
    lines.append("- real에서 ACF lag2/lag3가 크고, bernoulli에서 0 근처면 ‘짧은 래그 구조’가 데이터에 존재.")
    lines.append("- H(real)≈H(phase)이면 ‘PSD 모양(2차 통계)’ 성분이 핵심.")
    lines.append("- b_stage에서 약한데 z_stage에서 강해지면 ‘잔차화/정규화가 고주파를 강조’ 가능성.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


def _prime_factors_small(n: int) -> set[int]:
    x = int(n)
    out: set[int] = set()
    d = 2
    while d * d <= x:
        while x % d == 0:
            out.add(d)
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        out.add(x)
    return out


def _run_sieve_sweep(args: argparse.Namespace, Ws: list[int], is_prime: bytearray, out_path: Path) -> None:
    """
    7차-ish(원인 추적): sieve_mask_null을 단계적으로 강화하면서
    특정 f 피크들이 언제/어떤 소수 집합에서 생기는지 추적.

    - Dataset: sieve_mask_null only (deterministic).
    - Axis: k-axis (candidate index), consistent with prior observations.
    """
    if np is None:
        raise SystemExit("Sieve sweep mode requires numpy. Install with: python -m pip install numpy")

    sieve_primes = [int(x) for x in args.sieve_primes]
    steps = int(args.sieve_steps) if int(args.sieve_steps) > 0 else len(sieve_primes)
    steps = max(1, min(steps, len(sieve_primes)))
    tol = float(args.sieve_tol)
    topk = int(args.sieve_topk)
    # Targets (from your observed overlaps)
    targets = [0.397728, 0.488640, 0.432693, 0.410713]

    L = int(args.L)  # reuse global L
    window_name = str(args.windows[0]).lower() if args.windows else "hann"
    if window_name not in {"hann", "blackman", "rect"}:
        window_name = "hann"

    lines: list[str] = []
    lines.append(f"# Sieve sweep (cause tracing)  — N={args.N}, M={args.M}")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py --mode sieve_sweep` 실행 결과다.")
    lines.append("")
    lines.append("## 설정")
    lines.append("")
    lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
    lines.append(f"- sieve_primes: `{', '.join(map(str, sieve_primes))}` / steps: `{steps}`")
    lines.append(f"- tracking: L=`{L}`, window=`{window_name}`, num_windows=`{args.num_windows}`, K=`{args.K}`, delta_bins=`{args.track_delta_bins}`")
    lines.append(f"- perms: `{args.perms}` / shuffle_block: `{args.shuffle_block}` / seed: `{args.seed}`")
    lines.append(f"- overlap tol(f): `{tol}` / top stable peaks: `{topk}`")
    lines.append(f"- targets: `{', '.join([f'{t:.6f}' for t in targets])}`")
    lines.append("")

    def _lcm_many(nums: list[int]) -> int:
        v = 1
        for x in nums:
            v = math.lcm(int(v), int(x))
        return int(v)

    def _phi_from_factors(n: int) -> int:
        """Euler phi via prime factors (n is squarefree in our sweep)."""
        x = int(n)
        phi = x
        for p in _prime_factors_small(x):
            phi = phi // p * (p - 1)
        return int(phi)

    def _factor_str_small(n: int) -> str:
        """Partial factorization by small primes; remainder shown as rem(x)."""
        x = int(abs(n))
        if x == 0:
            return "0"
        if x == 1:
            return "1"
        parts: list[str] = []
        small_primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        ]
        for p in small_primes:
            if x % p != 0:
                continue
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            parts.append(f"{p}^{e}" if e > 1 else str(p))
            if x == 1:
                break
        if x != 1:
            parts.append(f"rem({x})")
        return "·".join(parts) if parts else str(n)

    def _pk_from_period(W: int, primes: list[int], Pn: int) -> int:
        """
        Exact count of k-samples in one n-period:
        Pk = #{1<=n<=Pn : gcd(n,W)=1 and n not divisible by any primes}
        Because Pn is a multiple of all moduli, the count is exact:
        Pk = Pn * Π_{p|W} (1-1/p) * Π_{q in primes} (1-1/q)
        """
        from fractions import Fraction

        frac = Fraction(Pn, 1)
        for p in _prime_factors_small(W):
            frac *= Fraction(p - 1, p)
        for q in primes:
            if q in _prime_factors_small(W):
                continue
            frac *= Fraction(q - 1, q)
        if frac.denominator != 1:
            # should not happen when Pn is lcm(W, primes)
            return int(frac.numerator // frac.denominator)
        return int(frac.numerator)

    def _cluster_freqs(items: list[dict[str, object]], tol_f: float) -> list[dict[str, object]]:
        # items: {"f": float, "step": int, "primes": str}
        clusters: list[dict[str, object]] = []
        for it in items:
            f = float(it["f"])
            placed = False
            for cl in clusters:
                if abs(f - float(cl["center_f"])) <= tol_f:
                    cl_items = cl["items"]
                    assert isinstance(cl_items, list)
                    cl_items.append(it)
                    cl["center_f"] = float(np.median([float(x["f"]) for x in cl_items]))
                    placed = True
                    break
            if not placed:
                clusters.append({"center_f": f, "items": [it]})
        clusters.sort(key=lambda c: len(c["items"]), reverse=True)  # type: ignore[index]
        return clusters

    def _grav_cluster_energy(stable_peaks: list[dict[str, object]], eps: float = 1e-12) -> tuple[float, float]:
        """
        7차 은유를 숫자로: 각 피크를 '질량'으로 보고, f-거리로 중력 포텐셜 유사량 계산.
        - mass := persistence * prom_mean  (prom_mean = peak/median PSD in that window)
        - energy := sum_{i<j} m_i m_j / (df^2 + eps)
        또한, 인접 피크 간격의 중앙값(median adjacent df)도 같이 반환.
        """
        if np is None:
            raise RuntimeError("numpy is required.")
        if not stable_peaks:
            return 0.0, float("nan")

        fs: list[float] = []
        ms: list[float] = []
        for d in stable_peaks:
            f = float(d.get("f", 0.0))
            persistence = float(d.get("persistence", 0.0))
            prom = float(d.get("prom_mean", 1.0))
            fs.append(f)
            ms.append(max(0.0, persistence * prom))

        E = 0.0
        n = len(fs)
        for i in range(n):
            for j in range(i + 1, n):
                df = abs(fs[i] - fs[j])
                E += (ms[i] * ms[j]) / (df * df + eps)

        fs_sorted = sorted(fs)
        adj = [fs_sorted[i + 1] - fs_sorted[i] for i in range(len(fs_sorted) - 1)]
        med_adj = float(np.median(np.array(adj, dtype=np.float64))) if adj else float("nan")
        return float(E), med_adj

    for W in Ws:
        lines.append(f"## W = {W}")
        lines.append("")
        banned = _prime_factors_small(W)
        use_primes = [q for q in sieve_primes if q not in banned]
        if not use_primes:
            lines.append("- (스윕 가능한 sieve_primes가 없음: W의 소인수와 전부 겹침)")
            lines.append("")
            continue

        cands_all = build_candidates(int(args.M), int(W))
        cand_n = np.frombuffer(cands_all, dtype=np.uint32).astype(np.int64, copy=True)

        step_results: list[dict[str, object]] = []
        all_peak_items: list[dict[str, object]] = []

        for i in range(1, min(steps, len(use_primes)) + 1):
            primes_i = use_primes[:i]
            primes_str = "{" + ",".join(map(str, primes_i)) + "}"

            Pn = _lcm_many([int(W)] + [int(q) for q in primes_i])
            Pk = _pk_from_period(int(W), primes_i, int(Pn))
            phiPn = _phi_from_factors(int(Pn))
            f0 = 1.0 / float(Pk) if Pk > 0 else float("nan")

            mask = np.ones(cand_n.shape[0], dtype=bool)
            for q in primes_i:
                mask &= (cand_n % int(q)) != 0
            b = mask.astype(np.float64)
            z = _standardize_np(b)

            stats_real, stable, score_real, _mean, _std, _maxv, p_val = tracking_with_permutation(
                z,
                perms=int(args.perms),
                shuffle_block=int(args.shuffle_block),
                seed=int(args.seed + (int(W) * 1000003) + i * 101),
                num_windows=int(args.num_windows),
                L=int(L),
                overlap=0.5,
                window_name=window_name,
                K=int(args.K),
                f_min_ratio=float(args.f_min_ratio),
                delta_bins=int(args.track_delta_bins),
            )

            stable_top = stable[:topk]
            freqs = [float(d.get("f", float(int(d["center_bin"]) / float(L)))) for d in stable_top]  # type: ignore[index]
            grav_E, med_df = _grav_cluster_energy(stable_top)  # type: ignore[arg-type]

            hit = {}
            harm = {}
            for t in targets:
                ok = any(abs(float(f) - float(t)) <= tol for f in freqs)
                hit[f"{t:.6f}"] = ok
                h = int(round(float(t) * float(Pk))) if Pk > 0 else 0
                approx = (h / float(Pk)) if Pk > 0 else float("nan")
                err = abs(float(t) - approx) if Pk > 0 else float("nan")
                g = math.gcd(int(h), int(Pk)) if Pk > 0 else 1
                rn = int(h // g) if g else int(h)
                rd = int(Pk // g) if g else int(Pk)
                harm[f"{t:.6f}"] = {"h": h, "approx": approx, "err": err}
                harm[f"{t:.6f}"].update({"g": g, "rn": rn, "rd": rd})

            step_results.append(
                {
                    "step": i,
                    "primes": primes_str,
                    "score": float(score_real),
                    "p": float(p_val),
                    "persist_70": int(stats_real["persist_70"]),
                    "drift": float(stats_real["mean_drift_bins"]),
                    "grav_E": float(grav_E),
                    "med_df": float(med_df),
                    "Pn": int(Pn),
                    "Pk": int(Pk),
                    "phiPn": int(phiPn),
                    "f0": float(f0),
                    "freqs": freqs,
                    "hit": hit,
                    "harm": harm,
                }
            )

            for f in freqs:
                all_peak_items.append({"f": float(f), "step": i, "primes": primes_str})

        lines.append("### 단계별 요약")
        lines.append("")
        lines.append("| step | primes | Pn | phi(Pn) | Pk | f0 | score | p | persist_70 | drift | target hits | top f (first 6) |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for r in step_results:
            hits = ", ".join([k for k, v in r["hit"].items() if v])  # type: ignore[union-attr]
            hits = hits if hits else "-"
            f6 = ", ".join([f"{x:.6f}" for x in list(r["freqs"])[:6]])  # type: ignore[arg-type]
            lines.append(
                f"| {r['step']} | {r['primes']} | {r['Pn']} | {r['phiPn']} | {r['Pk']} | {r['f0']:.3e} | {r['score']:.3f} | {r['p']:.4g} | {r['persist_70']} | {r['drift']:.3f} | {hits} | {f6} |"
            )
        lines.append("")

        lines.append("### 타겟 f의 '고조파 적합'(h≈f·Pk)")
        lines.append("")
        lines.append("| step | primes | target f | h | h/Pk | |f-h/Pk| | gcd(h,Pk) | reduced h/Pk | hit? |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---|---|")
        for r in step_results:
            for t in targets:
                key = f"{t:.6f}"
                hh = r["harm"][key]["h"]  # type: ignore[index]
                approx = r["harm"][key]["approx"]  # type: ignore[index]
                err = r["harm"][key]["err"]  # type: ignore[index]
                g = r["harm"][key]["g"]  # type: ignore[index]
                rn = r["harm"][key]["rn"]  # type: ignore[index]
                rd = r["harm"][key]["rd"]  # type: ignore[index]
                hit_flag = "Y" if r["hit"].get(key) else ""  # type: ignore[union-attr]
                lines.append(
                    f"| {r['step']} | {r['primes']} | {key} | {hh} | {approx:.6f} | {err:.3e} | {g} | {rn}/{rd} | {hit_flag} |"
                )
        lines.append("")

        # Reduced fraction stability: when rn/rd stops changing across steps.
        lines.append("### 약분 분수 안정화(왜 이 h인가? 1차 단서)")
        lines.append("")
        lines.append("| target f | stable from step | final reduced h/Pk | rd factors (small) |")
        lines.append("|---:|---:|---:|---|")
        for t in targets:
            key = f"{t:.6f}"
            seq: list[tuple[int, str, int]] = []
            for r in step_results:
                rn = int(r["harm"][key]["rn"])  # type: ignore[index]
                rd = int(r["harm"][key]["rd"])  # type: ignore[index]
                seq.append((int(r["step"]), f"{rn}/{rd}", rd))
            # Find last change index; if none, stable from first step.
            stable_from = seq[0][0] if seq else -1
            if seq:
                last_val = seq[-1][1]
                # stable_from is first step index such that all later values equal last_val
                for j in range(len(seq)):
                    if all(seq[k][1] == last_val for k in range(j, len(seq))):
                        stable_from = seq[j][0]
                        break
                final_rd = seq[-1][2]
                lines.append(f"| {key} | {stable_from} | {last_val} | {_factor_str_small(final_rd)} |")
        lines.append("")

        lines.append("### 피크 클러스터(단계 간 겹침, top f 기준)")
        lines.append("")
        clusters = _cluster_freqs(all_peak_items, tol_f=tol)
        lines.append("| center_f | support(steps) | steps |")
        lines.append("|---:|---:|---|")
        for cl in clusters[:25]:
            items = cl["items"]
            assert isinstance(items, list)
            steps_list = sorted({int(it["step"]) for it in items})
            lines.append(f"| {float(cl['center_f']):.6f} | {len(steps_list)} | {', '.join(map(str, steps_list))} |")
        lines.append("")

        lines.append("### 타겟 f가 처음 나타난 step")
        lines.append("")
        lines.append("| target f | first step | primes |")
        lines.append("|---:|---:|---|")
        for t in targets:
            key = f"{t:.6f}"
            first = next((r for r in step_results if r["hit"].get(key)), None)  # type: ignore[union-attr]
            if first is None:
                lines.append(f"| {key} | - | - |")
            else:
                lines.append(f"| {key} | {first['step']} | {first['primes']} |")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


def _run_mask_fft(args: argparse.Namespace, Ws: list[int], out_path: Path) -> None:
    """
    9차(why this h): 한 주기(Pn) 안에서 wheel 후보(k-축) 마스크 b(k)를 '정확히' 구성하고,
    그 1-주기 시퀀스의 DFT bin(h)가 타겟 주파수를 직접 예측하는지 본다.

    - period-n: Pn = lcm(W, sieve_primes[:step])
    - period-k(wheel): Pwheel = #{1..Pn : gcd(n,W)=1}
    - b(k)=1 if n not divisible by sieve primes, else 0   (단, wheel 후보에 대해서만)
    - 분석: DFT(|FFT|^2) 상위 bin들과 타겟들의 h_target=round(f*Pwheel) 일치 여부
    """
    if np is None:
        raise SystemExit("mask_fft mode requires numpy. Install with: python -m pip install numpy")

    sieve_primes = [int(x) for x in args.sieve_primes]
    steps = int(args.sieve_steps) if int(args.sieve_steps) > 0 else len(sieve_primes)
    steps = max(1, min(steps, len(sieve_primes)))
    topk = int(args.mask_topk)
    f_min_ratio = float(args.f_min_ratio)
    mask_max_pn = int(args.mask_max_pn)
    do_ablate = bool(getattr(args, "mask_ablate", False))
    do_swap_last = bool(getattr(args, "mask_swap_last", False))
    blur_block = int(getattr(args, "mask_blur_block", 0))
    blur_sweep_in = getattr(args, "mask_blur_sweep", None)
    blur_sweep = [int(x) for x in blur_sweep_in] if blur_sweep_in else []

    targets = [0.397728, 0.488640, 0.432693, 0.410713]

    lines: list[str] = []
    lines.append(f"# Mask FFT (why this h) — period-DFT, steps={steps}")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py --mode mask_fft` 실행 결과다.")
    lines.append("")
    lines.append("## 설정")
    lines.append("")
    lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
    lines.append(f"- sieve_primes: `{', '.join(map(str, sieve_primes))}` / steps: `{steps}`")
    lines.append(f"- mask_max_pn: `{mask_max_pn}`")
    lines.append(f"- f_min_ratio(저주파 컷): `{f_min_ratio}`")
    lines.append(f"- topK(표시): `{topk}`")
    lines.append(f"- targets: `{', '.join([f'{t:.6f}' for t in targets])}`")
    if blur_sweep:
        lines.append(f"- blur sweep: `{', '.join(map(str, blur_sweep))}`")
    lines.append("")

    def _lcm_many(nums: list[int]) -> int:
        v = 1
        for x in nums:
            v = math.lcm(int(v), int(x))
        return int(v)

    def _wheel_list(W: int, Pn: int) -> "np.ndarray":
        # Small Pn only (bounded by mask_max_pn)
        xs: list[int] = []
        for n in range(1, int(Pn) + 1):
            if gcd(int(n), int(W)) == 1:
                xs.append(int(n))
        return np.asarray(xs, dtype=np.int64)

    def _rank_desc(values: "np.ndarray", idx: int) -> int:
        v = float(values[idx])
        # rank 1 = largest
        return int(1 + int(np.sum(values > v)))

    def _blur_within_blocks(b: "np.ndarray", block: int, rng: "np.random.Generator") -> "np.ndarray":
        if block <= 1:
            return b
        n = int(b.shape[0])
        out = b.copy()
        for s in range(0, n, block):
            e = min(n, s + block)
            idx = rng.permutation(e - s)
            out[s:e] = out[s:e][idx]
        return out

    def _metrics_for_primes(
        W: int,
        Pn: int,
        wheel_n: "np.ndarray",
        primes_i: list[int],
        label: str,
        blur_block: int = 0,
        blur_seed: int = 0,
    ) -> dict[str, object]:
        Pwheel = int(wheel_n.shape[0])
        b = np.ones(Pwheel, dtype=np.float64)
        for q in primes_i:
            b *= ((wheel_n % int(q)) != 0).astype(np.float64)
        ones = int(np.sum(b))
        if blur_block and blur_block > 1:
            rng = np.random.default_rng(int(blur_seed))
            b = _blur_within_blocks(b, int(blur_block), rng)

        x = _standardize_np(b)
        fft = np.fft.rfft(x)
        psd = (np.abs(fft) ** 2).astype(np.float64)
        psd_no_dc = psd[1:] if psd.shape[0] > 1 else psd
        psd_med = float(np.median(psd_no_dc)) if psd_no_dc.size else float("nan")

        # peaks for "in topK" check
        peak_bins = pick_peaks(psd, K=int(topk), f_min_ratio=f_min_ratio)
        peak_bins = [int(k) for k in peak_bins if int(k) > 0]
        peak_bins_sorted = sorted(peak_bins, key=lambda k: float(psd[int(k)]), reverse=True)

        trows: list[dict[str, object]] = []
        for t in targets:
            h_star = int(round(float(t) * float(Pwheel)))
            f_hat = (float(h_star) / float(Pwheel)) if Pwheel > 0 else float("nan")
            err = abs(float(t) - f_hat) if Pwheel > 0 else float("nan")
            psd_h = float(psd[int(h_star)]) if 0 <= int(h_star) < int(psd.shape[0]) else float("nan")
            ratio_med = (psd_h / psd_med) if (math.isfinite(psd_h) and math.isfinite(psd_med) and psd_med > 0) else float("nan")
            rank_all = _rank_desc(psd[1:], max(0, h_star - 1)) if h_star > 0 else 10**9
            in_top = bool(h_star in peak_bins_sorted[:topk])
            trows.append(
                {
                    "t": float(t),
                    "h": int(h_star),
                    "f_hat": float(f_hat),
                    "err": float(err),
                    "psd": float(psd_h),
                    "ratio_med": float(ratio_med),
                    "rank": int(rank_all),
                    "in_top": bool(in_top),
                }
            )

        return {
            "label": label,
            "W": int(W),
            "Pn": int(Pn),
            "Pwheel": int(Pwheel),
            "ones": int(ones),
            "psd_med": float(psd_med),
            "targets": trows,
        }

    for W in Ws:
        lines.append(f"## W = {W}")
        lines.append("")
        banned = _prime_factors_small(int(W))
        use_primes = [q for q in sieve_primes if q not in banned]
        if not use_primes:
            lines.append("- (스윕 가능한 sieve_primes가 없음: W의 소인수와 전부 겹침)")
            lines.append("")
            continue

        for i in range(1, min(steps, len(use_primes)) + 1):
            primes_i = use_primes[:i]
            primes_str = "{" + ",".join(map(str, primes_i)) + "}"
            Pn = _lcm_many([int(W)] + [int(q) for q in primes_i])
            if Pn > mask_max_pn:
                lines.append(f"### step {i} / primes={primes_str}")
                lines.append("")
                lines.append(f"- skip: Pn={Pn} > mask_max_pn={mask_max_pn}")
                lines.append("")
                continue

            wheel_n = _wheel_list(int(W), int(Pn))
            Pwheel = int(wheel_n.shape[0])

            lines.append(f"### step {i} / primes={primes_str}")
            lines.append("")
            lines.append(f"- Pn: `{Pn}`")
            lines.append(f"- Pwheel(=#{'{1..Pn : gcd(n,W)=1}' }): `{Pwheel}`")
            lines.append("")

            # Baseline
            base = _metrics_for_primes(
                W=int(W),
                Pn=int(Pn),
                wheel_n=wheel_n,
                primes_i=primes_i,
                label="baseline",
            )
            lines.append(f"- ones(=survive primes_i among wheel): `{int(base['ones'])}`")
            lines.append(f"- PSD median(no-DC): `{float(base['psd_med']):.3e}`")
            if blur_block and blur_block > 1:
                blur = _metrics_for_primes(
                    W=int(W),
                    Pn=int(Pn),
                    wheel_n=wheel_n,
                    primes_i=primes_i,
                    label=f"blur(block={blur_block})",
                    blur_block=int(blur_block),
                    blur_seed=int(args.seed + (int(W) * 1000003) + int(Pn) + i * 17),
                )
                lines.append("")
                lines.append("#### boundary blur (local shuffle within blocks)")
                lines.append("")
                lines.append("| target f | PSD/med (baseline) | PSD/med (blur) | ratio(blur/base) | rank(base) | rank(blur) |")
                lines.append("|---:|---:|---:|---:|---:|---:|")
                bt = {f"{float(d['t']):.6f}": d for d in base["targets"]}  # type: ignore[index]
                zt = {f"{float(d['t']):.6f}": d for d in blur["targets"]}  # type: ignore[index]
                for t in targets:
                    k = f"{t:.6f}"
                    r_base = float(bt[k]["ratio_med"])
                    r_blur = float(zt[k]["ratio_med"])
                    rb = float(r_blur / r_base) if (math.isfinite(r_blur) and math.isfinite(r_base) and r_base > 0) else float("nan")
                    lines.append(
                        f"| {k} | {r_base:.3e} | {r_blur:.3e} | {rb:.3e} | {int(bt[k]['rank'])} | {int(zt[k]['rank'])} |"
                    )
                lines.append("")

            if blur_sweep:
                lines.append("#### blur sweep (which boundary scale matters?)")
                lines.append("")
                lines.append("| blur_block | target f | PSD/med (blur) | ratio(blur/base) | rank(blur) |")
                lines.append("|---:|---:|---:|---:|---:|")
                bt = {f"{float(d['t']):.6f}": d for d in base["targets"]}  # type: ignore[index]
                # Accumulate for summary: per target, track min/max ratio across sweep
                sweep_stats: dict[str, dict[str, float | int]] = {}
                for t in targets:
                    k = f"{t:.6f}"
                    sweep_stats[k] = {
                        "min_ratio": float("inf"),
                        "min_bb": -1,
                        "min_psdmed": float("nan"),
                        "min_rank": -1,
                        "max_ratio": -float("inf"),
                        "max_bb": -1,
                        "max_psdmed": float("nan"),
                        "max_rank": -1,
                    }
                for bb in blur_sweep:
                    if int(bb) <= 1:
                        continue
                    blur = _metrics_for_primes(
                        W=int(W),
                        Pn=int(Pn),
                        wheel_n=wheel_n,
                        primes_i=primes_i,
                        label=f"blur(block={bb})",
                        blur_block=int(bb),
                        blur_seed=int(args.seed + (int(W) * 1000003) + int(Pn) + i * 17 + int(bb) * 1009),
                    )
                    zt = {f"{float(d['t']):.6f}": d for d in blur["targets"]}  # type: ignore[index]
                    for t in targets:
                        k = f"{t:.6f}"
                        r_base = float(bt[k]["ratio_med"])
                        r_blur = float(zt[k]["ratio_med"])
                        rb = float(r_blur / r_base) if (math.isfinite(r_blur) and math.isfinite(r_base) and r_base > 0) else float("nan")
                        lines.append(f"| {bb} | {k} | {r_blur:.3e} | {rb:.3e} | {int(zt[k]['rank'])} |")
                        if math.isfinite(rb):
                            cur = sweep_stats[k]
                            if float(rb) < float(cur["min_ratio"]):
                                cur["min_ratio"] = float(rb)
                                cur["min_bb"] = int(bb)
                                cur["min_psdmed"] = float(r_blur)
                                cur["min_rank"] = int(zt[k]["rank"])
                            if float(rb) > float(cur["max_ratio"]):
                                cur["max_ratio"] = float(rb)
                                cur["max_bb"] = int(bb)
                                cur["max_psdmed"] = float(r_blur)
                                cur["max_rank"] = int(zt[k]["rank"])
                lines.append("")
                lines.append("#### blur sweep summary (min/max ratio per target)")
                lines.append("")
                lines.append("| target f | base PSD/med | min ratio | at block | blur PSD/med | rank | max ratio | at block | blur PSD/med | rank |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for t in targets:
                    k = f"{t:.6f}"
                    r_base = float(bt[k]["ratio_med"])
                    st = sweep_stats[k]
                    lines.append(
                        f"| {k} | {r_base:.3e} | {float(st['min_ratio']):.3e} | {int(st['min_bb'])} | {float(st['min_psdmed']):.3e} | {int(st['min_rank'])} | {float(st['max_ratio']):.3e} | {int(st['max_bb'])} | {float(st['max_psdmed']):.3e} | {int(st['max_rank'])} |"
                    )
                lines.append("")

            if do_ablate and len(primes_i) >= 2:
                lines.append("#### rune ablation (drop one prime)")
                lines.append("")
                lines.append("| dropped | target f | PSD/med | ratio(vs baseline) | rank | in topK? |")
                lines.append("|---:|---:|---:|---:|---:|---|")
                bt = {f"{float(d['t']):.6f}": d for d in base["targets"]}  # type: ignore[index]
                for drop in primes_i:
                    primes_drop = [q for q in primes_i if q != drop]
                    v = _metrics_for_primes(
                        W=int(W),
                        Pn=int(Pn),
                        wheel_n=wheel_n,
                        primes_i=primes_drop,
                        label=f"drop {drop}",
                    )
                    vt = {f"{float(d['t']):.6f}": d for d in v["targets"]}  # type: ignore[index]
                    for t in targets:
                        k = f"{t:.6f}"
                        r_base = float(bt[k]["ratio_med"])
                        r_v = float(vt[k]["ratio_med"])
                        rvb = float(r_v / r_base) if (math.isfinite(r_v) and math.isfinite(r_base) and r_base > 0) else float("nan")
                        lines.append(
                            f"| {drop} | {k} | {r_v:.3e} | {rvb:.3e} | {int(vt[k]['rank'])} | {'Y' if bool(vt[k]['in_top']) else ''} |"
                        )
                lines.append("")

            if do_swap_last and (i < len(use_primes)):
                # swap the last included prime with next unused prime
                swap_out = primes_i[-1]
                swap_in = use_primes[i]
                primes_swap = primes_i[:-1] + [swap_in]
                lines.append("#### rune swap (swap last with next)")
                lines.append("")
                lines.append(f"- swap: `{swap_out}` -> `{swap_in}`")
                lines.append("")
                v = _metrics_for_primes(
                    W=int(W),
                    Pn=int(Pn),
                    wheel_n=wheel_n,
                    primes_i=primes_swap,
                    label=f"swap {swap_out}->{swap_in}",
                )
                bt = {f"{float(d['t']):.6f}": d for d in base["targets"]}  # type: ignore[index]
                vt = {f"{float(d['t']):.6f}": d for d in v["targets"]}  # type: ignore[index]
                lines.append("| target f | PSD/med (base) | PSD/med (swap) | ratio(swap/base) | rank(base) | rank(swap) |")
                lines.append("|---:|---:|---:|---:|---:|---:|")
                for t in targets:
                    k = f"{t:.6f}"
                    r_base = float(bt[k]["ratio_med"])
                    r_v = float(vt[k]["ratio_med"])
                    rvb = float(r_v / r_base) if (math.isfinite(r_v) and math.isfinite(r_base) and r_base > 0) else float("nan")
                    lines.append(
                        f"| {k} | {r_base:.3e} | {r_v:.3e} | {rvb:.3e} | {int(bt[k]['rank'])} | {int(vt[k]['rank'])} |"
                    )
                lines.append("")

            lines.append("#### top DFT bins (mask period)")
            lines.append("")
            lines.append("| rank | h(bin) | f=h/Pwheel | PSD |")
            lines.append("|---:|---:|---:|---:|")
            # Recompute baseline PSD for top-bin display
            b = np.ones(Pwheel, dtype=np.float64)
            for q in primes_i:
                b *= ((wheel_n % int(q)) != 0).astype(np.float64)
            x = _standardize_np(b)
            fft = np.fft.rfft(x)
            psd = (np.abs(fft) ** 2).astype(np.float64)
            peak_bins = pick_peaks(psd, K=int(topk), f_min_ratio=f_min_ratio)
            peak_bins = [int(k) for k in peak_bins if int(k) > 0]
            peak_bins_sorted = sorted(peak_bins, key=lambda k: float(psd[int(k)]), reverse=True)
            for rnk, h in enumerate(peak_bins_sorted[:topk], start=1):
                f = float(h) / float(Pwheel)
                lines.append(f"| {rnk} | {h} | {f:.6f} | {float(psd[int(h)]):.3e} |")
            lines.append("")

            lines.append("#### targets -> predicted h (period-k wheel)")
            lines.append("")
            lines.append("| target f | h*=round(f·Pwheel) | h*/Pwheel | |f-h*/Pwheel| | PSD(h*) | PSD/med | rank(PSD) | in topK? |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
            bt = {f"{float(d['t']):.6f}": d for d in base["targets"]}  # type: ignore[index]
            for t in targets:
                k = f"{t:.6f}"
                d = bt[k]
                lines.append(
                    f"| {k} | {int(d['h'])} | {float(d['f_hat']):.6f} | {float(d['err']):.3e} | {float(d['psd']):.3e} | {float(d['ratio_med']):.3e} | {int(d['rank'])} | {'Y' if bool(d['in_top']) else ''} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")

@dataclass(frozen=True)
class ValidationTrackingResult:
    dataset: str
    W: int
    axis: str
    window: str
    L: int
    window_len: int
    clusters: int
    persist_70: int
    persist_50: int
    mean_drift_bins: float
    score_real: float
    prom_score_real: float
    score_perm_mean: float
    score_perm_std: float
    score_perm_max: float
    p_value: float
    prom_score_perm_mean: float
    prom_score_perm_std: float
    prom_score_perm_max: float
    prom_p_value: float
    stable_peaks: list[dict[str, float | int]]


def _build_p_for_candidates(cands: array, A: float) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    p = np.empty(len(cands), dtype=np.float64)
    for i, n in enumerate(cands):
        p[i] = _clamp_prob(A / math.log(int(n)))
    return p


def _build_z_from_b_p(b: "np.ndarray", p: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    p2 = np.clip(p, EPS, 1.0 - EPS)
    denom = np.sqrt(np.maximum(EPS, p2 * (1.0 - p2)))
    return (b - p2) / denom


def axis_k(z: "np.ndarray") -> "np.ndarray":
    return z


def axis_n(candidate_n: "np.ndarray", z: "np.ndarray", M: int) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    arr = np.zeros(M + 1, dtype=np.float64)
    arr[candidate_n.astype(np.int64)] = z
    return arr[2:]


def axis_logn(candidate_n: "np.ndarray", z: "np.ndarray", num_points: int | None = None) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required.")
    x = np.log(candidate_n.astype(np.float64))
    y = z.astype(np.float64, copy=False)
    if num_points is None:
        num_points = int(len(z))
    x_new = np.linspace(float(x.min()), float(x.max()), int(num_points))
    y_new = np.interp(x_new, x, y)
    return y_new


def phase_randomize_real(
    x: "np.ndarray",
    *,
    seed: int,
    keep_dc: bool = True,
    keep_nyquist: bool = True,
) -> "np.ndarray":
    """
    Phase-randomized surrogate for real-valued signal x (rFFT-based).
    Preserves per-bin magnitude (thus PSD shape), destroys temporal correlations.
    """
    if np is None:
        raise RuntimeError("numpy is required.")
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    if n < 4:
        return x.copy()

    X = np.fft.rfft(x)
    mag = np.abs(X)

    phases = rng.uniform(0.0, 2.0 * np.pi, size=mag.shape[0])
    Y = mag * np.exp(1j * phases)

    if keep_dc:
        Y[0] = X[0]
    if keep_nyquist and (n % 2 == 0) and (Y.shape[0] > 1):
        Y[-1] = X[-1]

    y = np.fft.irfft(Y, n=n).astype(np.float64)

    # normalize to same mean/std (recommended for fair comparison under same metric)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if y_std > 0 and x_std > 0:
        y = y - float(np.mean(y))
        y = y * (x_std / y_std)
        y = y + float(np.mean(x))

    return y


def _run_tracking_experiment_on_z(
    z: "np.ndarray",
    *,
    dataset: str,
    W: int,
    axis: str,
    L: int,
    window_name: str,
    perms: int,
    shuffle_block: int,
    seed: int,
    num_windows: int,
    K: int,
    f_min_ratio: float,
    delta_bins: int,
) -> ValidationTrackingResult:
    stats_real, stable, score_real, mean, std, maxv, p_value = tracking_with_permutation(
        z,
        perms=perms,
        shuffle_block=shuffle_block,
        seed=seed,
        num_windows=num_windows,
        L=L,
        overlap=0.5,
        window_name=window_name,
        K=K,
        f_min_ratio=f_min_ratio,
        delta_bins=delta_bins,
    )
    return ValidationTrackingResult(
        dataset=dataset,
        W=W,
        axis=axis,
        window=window_name,
        L=L,
        window_len=int(stats_real["window_len"]),
        clusters=int(stats_real["num_clusters"]),
        persist_70=int(stats_real["persist_70"]),
        persist_50=int(stats_real["persist_50"]),
        mean_drift_bins=float(stats_real["mean_drift_bins"]),
        score_real=float(score_real),
        prom_score_real=float(stats_real.get("prom_score_real", float("nan"))),
        score_perm_mean=float(mean),
        score_perm_std=float(std),
        score_perm_max=float(maxv),
        p_value=float(p_value),
        prom_score_perm_mean=float(stats_real.get("prom_score_perm_mean", float("nan"))),
        prom_score_perm_std=float(stats_real.get("prom_score_perm_std", float("nan"))),
        prom_score_perm_max=float(stats_real.get("prom_score_perm_max", float("nan"))),
        prom_p_value=float(stats_real.get("prom_p_value", float("nan"))),
        stable_peaks=[{k: v for k, v in d.items()} for d in stable[:15]],
    )


def _find_frequency_overlap(
    results: list[ValidationTrackingResult],
    *,
    top_k: int = 10,
    tol: float = 5e-4,
) -> list[dict[str, object]]:
    if np is None:
        raise RuntimeError("numpy is required.")
    real_results = [r for r in results if r.dataset == "real"]
    items: list[dict[str, object]] = []
    for r in real_results:
        for peak in r.stable_peaks[:top_k]:
            items.append(
                {
                    "W": r.W,
                    "axis": r.axis,
                    "window": r.window,
                    "L": r.L,
                    "f": float(peak.get("f", float(int(peak["center_bin"]) / float(r.L)))),  # type: ignore[index]
                    "persistence": float(peak["persistence"]),
                }
            )

    clusters: list[dict[str, object]] = []
    for it in items:
        f = float(it["f"])
        placed = False
        for cl in clusters:
            if abs(f - float(cl["center_f"])) <= tol:
                cl_items = cl["items"]
                assert isinstance(cl_items, list)
                cl_items.append(it)
                cl["center_f"] = float(np.median([float(x["f"]) for x in cl_items]))
                placed = True
                break
        if not placed:
            clusters.append({"center_f": f, "items": [it]})
    clusters.sort(key=lambda c: len(c["items"]), reverse=True)  # type: ignore[index]
    return clusters


def _run_validation(args: argparse.Namespace, Ws: list[int], is_prime: bytearray, out_path: Path) -> None:
    if np is None:
        raise SystemExit("Validation mode requires numpy. Install with: python -m pip install numpy")

    def _prime_factors(n: int) -> set[int]:
        x = int(n)
        out: set[int] = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                out.add(d)
                x //= d
            d += 1 if d == 2 else 2
        if x > 1:
            out.add(x)
        return out

    L_list = [int(x) for x in args.L_list]
    windows = [str(w).lower() for w in args.windows]
    axes = [str(a).lower() for a in args.axes]
    for w in windows:
        if w not in {"hann", "blackman", "rect"}:
            raise SystemExit(f"Unknown window '{w}'. Use hann/blackman/rect.")
    for a in axes:
        if a not in {"k", "n", "logn"}:
            raise SystemExit(f"Unknown axis '{a}'. Use k/n/logn.")

    lines: list[str] = []
    lines.append(f"# Riemann-ish validation pipeline (N={args.N}, M={args.M})")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py --mode validation` 실행 결과다.")
    lines.append("")
    lines.append("## 설정")
    lines.append("")
    lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
    lines.append(f"- L_list: `{', '.join(map(str, L_list))}`")
    lines.append(f"- windows: `{', '.join(windows)}`")
    lines.append(f"- axes: `{', '.join(axes)}`")
    lines.append(f"- num_windows: `{args.num_windows}` / K: `{args.K}` / f_min_ratio: `{args.f_min_ratio}` / delta_bins: `{args.track_delta_bins}`")
    lines.append(f"- perms: `{args.perms}` / shuffle_block: `{args.shuffle_block}` / seed: `{args.seed}`")
    lines.append("")

    results: list[ValidationTrackingResult] = []
    warnings: list[str] = []

    def _split_equal_windows_np(x: "np.ndarray", num_windows: int) -> tuple[list["np.ndarray"], int]:
        n = int(x.shape[0])
        win_len = n // int(num_windows)
        n_use = win_len * int(num_windows)
        x_use = x[:n_use]
        return [x_use[i * win_len : (i + 1) * win_len] for i in range(int(num_windows))], int(win_len)

    def _fixed_peaks_score(
        z: "np.ndarray",
        *,
        L: int,
        window_name: str,
        num_windows: int,
        fixed_bins: list[int],
        delta_bins: int,
        overlap: float = 0.5,
    ) -> float:
        """
        Evaluate dataset using peak bins learned from REAL only.
        For each k-window: compute PSD, then for each fixed bin measure local max ±delta
        and accumulate mean log-prominence (peak/median).
        """
        if np is None:
            raise RuntimeError("numpy is required.")
        windows, win_len = _split_equal_windows_np(z, num_windows=num_windows)
        if win_len < L:
            raise ValueError(f"window_len={win_len} < L={L}")
        s = 0.0
        for seg in windows:
            psd = welch_psd(seg, L=L, overlap=overlap, window_name=window_name)
            base = float(np.median(psd))
            base = max(base, 1e-30)
            nbin = int(psd.shape[0])
            for b in fixed_bins:
                lo = max(1, int(b) - int(delta_bins))
                hi = min(nbin - 1, int(b) + int(delta_bins) + 1)
                peak = float(np.max(psd[lo:hi]))
                prom = max(peak / base, 1e-30)
                s += math.log(prom)
        return float(s)

    for W in Ws:
        cands_all = build_candidates(args.M, W)
        y_all = array("b")
        for n in cands_all:
            y_all.append(1 if is_prime[n] else 0)

        # fit A on train (n<=N)
        c_train = array("I")
        y_train = array("b")
        for n, yi in zip(cands_all, y_all):
            if n <= args.N:
                c_train.append(n)
                y_train.append(yi)
            else:
                break

        A = fit_A_scale(c_train, y_train)
        cand_n = np.frombuffer(cands_all, dtype=np.uint32).astype(np.int64, copy=True)
        b_real = np.frombuffer(y_all, dtype=np.int8).astype(np.float64, copy=True)
        p = _build_p_for_candidates(cands_all, A)
        z_real_k = _build_z_from_b_p(b_real, p)

        rng = np.random.default_rng(args.seed + (W * 1000003) + 123)
        b_bern = rng.binomial(1, np.clip(p, EPS, 1.0 - EPS)).astype(np.float64)
        z_bern_k = _build_z_from_b_p(b_bern, p)
        z_shuffle_k = block_shuffle_np(z_real_k, block=int(args.shuffle_block), rng=rng)
        z_phase_k = phase_randomize_real(z_real_k, seed=int(args.seed + (W * 1000003) + 991))

        # (6차-B) sieve-mask control: deterministic compositeness mask by small primes (beyond wheel W)
        sieve_primes = [7, 11, 13, 17, 19, 23, 29]
        banned = _prime_factors(W)
        mask = np.ones(cand_n.shape[0], dtype=bool)
        for q in sieve_primes:
            if q in banned:
                continue
            mask &= (cand_n % q) != 0
        b_sieve = mask.astype(np.float64)
        z_sieve_k = _standardize_np(b_sieve)

        datasets: dict[str, "np.ndarray"] = {
            "real": z_real_k,
            "bernoulli_null": z_bern_k,
            "shuffle_null": z_shuffle_k,
            "phase_null": z_phase_k,
            "sieve_mask_null": z_sieve_k,
        }
        dataset_seed = {"real": 1, "bernoulli_null": 2, "shuffle_null": 3, "phase_null": 4, "sieve_mask_null": 5}

        for dataset_name, z0 in datasets.items():
            for axis_name in axes:
                if axis_name == "k":
                    z_axis = axis_k(z0)
                elif axis_name == "n":
                    z_axis = axis_n(cand_n, z0, M=args.M)
                elif axis_name == "logn":
                    z_axis = axis_logn(cand_n, z0, num_points=len(z0))
                else:
                    raise RuntimeError(axis_name)

                for L in L_list:
                    for window_name in windows:
                        try:
                            res = _run_tracking_experiment_on_z(
                                z_axis,
                                dataset=dataset_name,
                                W=W,
                                axis=axis_name,
                                L=L,
                                window_name=window_name,
                                perms=int(args.perms),
                                shuffle_block=int(args.shuffle_block),
                                seed=int(args.seed + (W * 1000003) + (dataset_seed[dataset_name] * 10007) + L),
                                num_windows=int(args.num_windows),
                                K=int(args.K),
                                f_min_ratio=float(args.f_min_ratio),
                                delta_bins=int(args.track_delta_bins),
                            )
                            results.append(res)
                        except Exception as e:
                            warnings.append(
                                f"skip dataset={dataset_name} W={W} axis={axis_name} L={L} window={window_name}: {e}"
                            )

    lines.append("## validation summary")
    lines.append("")
    lines.append("| dataset | W | axis | window | L | score_real | prom_score | p | prom_p | persist_70 | persist_50 | drift |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    results_sorted = sorted(results, key=lambda r: (r.dataset, r.W, r.axis, r.window, r.L))
    for r in results_sorted:
        lines.append(
            f"| {r.dataset} | {r.W} | {r.axis} | {r.window} | {r.L} | {r.score_real:.3f} | {r.prom_score_real:.3f} | {r.p_value:.4g} | {r.prom_p_value:.4g} | {r.persist_70} | {r.persist_50} | {r.mean_drift_bins:.3f} |"
        )
    lines.append("")

    lines.append("## real vs controls")
    lines.append("")
    lines.append("| W | axis | window | L | real_score | bernoulli_score | shuffle_score | phase_score |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    by_cfg: dict[tuple[int, str, str, int], dict[str, float]] = {}
    for r in results:
        key = (r.W, r.axis, r.window, r.L)
        by_cfg.setdefault(key, {})[r.dataset] = r.score_real
    for (W, axis_name, window_name, L), m in sorted(by_cfg.items()):
        lines.append(
            f"| {W} | {axis_name} | {window_name} | {L} | {m.get('real', float('nan')):.3f} | {m.get('bernoulli_null', float('nan')):.3f} | {m.get('shuffle_null', float('nan')):.3f} | {m.get('phase_null', float('nan')):.3f} |"
        )
    lines.append("")

    lines.append("## real vs phase_null")
    lines.append("")
    lines.append("| W | axis | window | L | real_score | phase_score | real_p | phase_p | real_prom | phase_prom | real_prom_p | phase_prom_p |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    # index p-values too
    by_cfg_p: dict[tuple[int, str, str, int], dict[str, float]] = {}
    by_cfg_prom: dict[tuple[int, str, str, int], dict[str, float]] = {}
    by_cfg_prom_p: dict[tuple[int, str, str, int], dict[str, float]] = {}
    for r in results:
        key = (r.W, r.axis, r.window, r.L)
        by_cfg_p.setdefault(key, {})[r.dataset] = r.p_value
        by_cfg_prom.setdefault(key, {})[r.dataset] = r.prom_score_real
        by_cfg_prom_p.setdefault(key, {})[r.dataset] = r.prom_p_value
    for (W, axis_name, window_name, L), m in sorted(by_cfg.items()):
        mp = by_cfg_p.get((W, axis_name, window_name, L), {})
        mpr = by_cfg_prom.get((W, axis_name, window_name, L), {})
        mpp = by_cfg_prom_p.get((W, axis_name, window_name, L), {})
        lines.append(
            f"| {W} | {axis_name} | {window_name} | {L} | {m.get('real', float('nan')):.3f} | {m.get('phase_null', float('nan')):.3f} | {mp.get('real', float('nan')):.4g} | {mp.get('phase_null', float('nan')):.4g} | {mpr.get('real', float('nan')):.3f} | {mpr.get('phase_null', float('nan')):.3f} | {mpp.get('real', float('nan')):.4g} | {mpp.get('phase_null', float('nan')):.4g} |"
        )
    lines.append("")

    lines.append("## real vs sieve_mask_null (wheel+small primes mask)")
    lines.append("")
    lines.append("| W | axis | window | L | real_score | sieve_score | real_p | sieve_p |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for (W, axis_name, window_name, L), m in sorted(by_cfg.items()):
        mp = by_cfg_p.get((W, axis_name, window_name, L), {})
        lines.append(
            f"| {W} | {axis_name} | {window_name} | {L} | {m.get('real', float('nan')):.3f} | {m.get('sieve_mask_null', float('nan')):.3f} | {mp.get('real', float('nan')):.4g} | {mp.get('sieve_mask_null', float('nan')):.4g} |"
        )
    lines.append("")

    lines.append("## frequency overlap (real only)")
    lines.append("")
    clusters = _find_frequency_overlap(results, top_k=10, tol=5e-4)
    lines.append("| center_f | support | examples |")
    lines.append("|---:|---:|---|")
    for cl in clusters[:20]:
        items = cl["items"]
        assert isinstance(items, list)
        ex = ", ".join([f"W={it['W']},axis={it['axis']},win={it['window']},L={it['L']}" for it in items[:6]])
        lines.append(f"| {float(cl['center_f']):.6f} | {len(items)} | {ex} |")
    lines.append("")

    # Fixed-peaks cross-dataset check (5차 핵심 분기 강화)
    lines.append("## fixed-peaks cross-dataset (peaks learned from REAL only)")
    lines.append("")
    lines.append(
        "각 config(W/axis/window/L)에서 REAL의 stable_peaks 상위 10개 center_bin을 고정한 뒤, "
        "다른 데이터셋에 그대로 적용해 PSD prominence 합(로그)을 계산한다."
    )
    lines.append("")
    lines.append("| W | axis | window | L | real_fixed | phase_fixed | shuffle_fixed | bernoulli_fixed |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")

    # Build quick index for z_axis signals per W/dataset/axis to reuse for fixed scoring
    z_axis_cache: dict[tuple[int, str, str], "np.ndarray"] = {}
    # Recompute minimal needed z for fixed scoring (axis=k only in 5차 기본, but support axes list)
    for W in Ws:
        cands_all = build_candidates(args.M, W)
        y_all = array("b")
        for n in cands_all:
            y_all.append(1 if is_prime[n] else 0)
        c_train = array("I")
        y_train = array("b")
        for n, yi in zip(cands_all, y_all):
            if n <= args.N:
                c_train.append(n)
                y_train.append(yi)
            else:
                break
        A = fit_A_scale(c_train, y_train)
        cand_n = np.frombuffer(cands_all, dtype=np.uint32).astype(np.int64, copy=True)
        b_real = np.frombuffer(y_all, dtype=np.int8).astype(np.float64, copy=True)
        p = _build_p_for_candidates(cands_all, A)
        z_real_k = _build_z_from_b_p(b_real, p)
        rng = np.random.default_rng(args.seed + (W * 1000003) + 123)
        b_bern = rng.binomial(1, np.clip(p, EPS, 1.0 - EPS)).astype(np.float64)
        z_bern_k = _build_z_from_b_p(b_bern, p)
        z_shuffle_k = block_shuffle_np(z_real_k, block=int(args.shuffle_block), rng=rng)
        z_phase_k = phase_randomize_real(z_real_k, seed=int(args.seed + (W * 1000003) + 991))

        base = {"real": z_real_k, "phase_null": z_phase_k, "shuffle_null": z_shuffle_k, "bernoulli_null": z_bern_k}
        for dataset_name, z0 in base.items():
            for axis_name in axes:
                if axis_name == "k":
                    z_axis = axis_k(z0)
                elif axis_name == "n":
                    z_axis = axis_n(cand_n, z0, M=args.M)
                elif axis_name == "logn":
                    z_axis = axis_logn(cand_n, z0, num_points=len(z0))
                else:
                    continue
                z_axis_cache[(W, dataset_name, axis_name)] = z_axis

    # Index real stable peaks by config
    real_peaks_by_cfg: dict[tuple[int, str, str, int], list[int]] = {}
    for r in results:
        if r.dataset != "real":
            continue
        key = (r.W, r.axis, r.window, r.L)
        fixed = [int(p["center_bin"]) for p in r.stable_peaks[:10] if "center_bin" in p]
        if fixed:
            real_peaks_by_cfg[key] = fixed

    for W in Ws:
        for axis_name in axes:
            for window_name in windows:
                for L in L_list:
                    key = (W, axis_name, window_name, L)
                    fixed_bins = real_peaks_by_cfg.get(key)
                    if not fixed_bins:
                        continue
                    try:
                        real_fixed = _fixed_peaks_score(
                            z_axis_cache[(W, "real", axis_name)],
                            L=L,
                            window_name=window_name,
                            num_windows=int(args.num_windows),
                            fixed_bins=fixed_bins,
                            delta_bins=int(args.track_delta_bins),
                        )
                        phase_fixed = _fixed_peaks_score(
                            z_axis_cache[(W, "phase_null", axis_name)],
                            L=L,
                            window_name=window_name,
                            num_windows=int(args.num_windows),
                            fixed_bins=fixed_bins,
                            delta_bins=int(args.track_delta_bins),
                        )
                        shuffle_fixed = _fixed_peaks_score(
                            z_axis_cache[(W, "shuffle_null", axis_name)],
                            L=L,
                            window_name=window_name,
                            num_windows=int(args.num_windows),
                            fixed_bins=fixed_bins,
                            delta_bins=int(args.track_delta_bins),
                        )
                        bern_fixed = _fixed_peaks_score(
                            z_axis_cache[(W, "bernoulli_null", axis_name)],
                            L=L,
                            window_name=window_name,
                            num_windows=int(args.num_windows),
                            fixed_bins=fixed_bins,
                            delta_bins=int(args.track_delta_bins),
                        )
                        lines.append(
                            f"| {W} | {axis_name} | {window_name} | {L} | {real_fixed:.3f} | {phase_fixed:.3f} | {shuffle_fixed:.3f} | {bern_fixed:.3f} |"
                        )
                    except Exception as e:
                        warnings.append(f"fixed-peaks skip W={W} axis={axis_name} L={L} window={window_name}: {e}")

    lines.append("")

    if warnings:
        lines.append("## WARN (skipped configs)")
        lines.append("")
        for w in warnings[:50]:
            lines.append(f"- {w}")
        if len(warnings) > 50:
            lines.append(f"- ... and {len(warnings) - 50} more")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


def _run_tracking(args: argparse.Namespace, Ws: list[int], is_prime: bytearray, out_path: Path) -> None:
    if np is None:
        raise SystemExit("Tracking mode requires numpy. Install with: python -m pip install numpy")

    lines: list[str] = []
    lines.append(f"# Riemann-ish peak tracking experiment (N={args.N}, M={args.M})")
    lines.append("")
    lines.append("이 문서는 `tools/riemann_experiment.py --mode tracking` 실행 결과다.")
    lines.append("")
    lines.append("## 설정")
    lines.append("")
    lines.append(f"- W 목록: `{', '.join(map(str, Ws))}`")
    lines.append(f"- num_windows: `{args.num_windows}` (k-구간 분할, B안)")
    lines.append(f"- Welch: L=`{args.L}`, overlap=`0.5`, window=`Hann`")
    lines.append(f"- peaks: K=`{args.K}`, f_min_ratio=`{args.f_min_ratio}`")
    lines.append(f"- clustering: delta_bins=`{args.track_delta_bins}`")
    lines.append(f"- perms: `{args.perms}` / shuffle_block: `{args.shuffle_block}`")
    lines.append("")

    lines.append("## 결과 요약")
    lines.append("")
    lines.append("| W | window_len | clusters | persist_70 | persist_50 | mean_drift (bins) | score_real | score_perm_mean | score_perm_max | p-value | elapsed (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    detailed: list[tuple[int, TrackingResult]] = []
    for W in Ws:
        r = tracking_experiment(
            N=args.N,
            M=args.M,
            is_prime=is_prime,
            W=W,
            L=args.L,
            overlap=0.5,
            K=args.K,
            f_min_ratio=args.f_min_ratio,
            delta_bins=args.track_delta_bins,
            num_windows=args.num_windows,
            perms=args.perms,
            shuffle_block=args.shuffle_block,
            seed=args.seed + (W * 1000003) + 29,
        )
        detailed.append((W, r))
        lines.append(
            f"| {r.W} | {r.window_len} | {r.num_clusters} | {r.persist_70} | {r.persist_50} | {r.mean_drift_bins:.3f} | {r.score_real:.3f} | {r.score_perm_mean:.3f} | {r.score_perm_max:.3f} | {r.p_value:.4g} | {r.elapsed_s:.2f} |"
        )

    lines.append("")
    lines.append("## 안정 피크 목록(상위 15)")
    lines.append("")
    lines.append("각 W에 대해 persistence(등장률) 높은 순으로 정렬된 클러스터 중심 bin을 적어둔다.")
    lines.append("")
    for W, r in detailed:
        lines.append(f"### W = {W}")
        lines.append("")
        lines.append("| center_bin | f=center_bin/L | period≈L/bin | persistence | drift_std | hits |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for d in r.stable_peaks:
            center = int(d["center_bin"])
            f = center / float(r.L)
            period = (float(r.L) / center) if center != 0 else float("inf")
            lines.append(
                f"| {center} | {f:.6f} | {period:.3f} | {float(d['persistence']):.3f} | {float(d['drift_std']):.3f} | {int(d['hits'])} |"
            )
        lines.append("")

    lines.append("## 판정 가이드(3차 모델)")
    lines.append("")
    lines.append("- persist_70 / persist_50가 perm 대비 유의하게 크고(p-value ↓), mean_drift가 낮으면 ‘고정 피크’ 쪽.")
    lines.append("- 반대로 persistent peaks가 거의 없고 drift가 크면 ‘떠도는 피크(착시/비정상성/채굴)’ 쪽.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

