# DistributionShift - Detect distribution shift between activation sets via spectra
# Part of the Kapildhinwa2010-dev Toolkit.

"""DeepAnalyze - layer-wise spectral analysis for deep networks.

Lineage: Reverse.Dean (b2/b4). Given a dict of weight matrices ("layers"),
this module computes singular-value spectra, effective ranks and condition
numbers, and ranks layers by bottleneck severity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "layer_spectrum",
    "effective_rank",
    "layer_report",
    "analyze_model",
    "find_bottlenecks",
    "spectrum_distance",
    "shift_report",
]


def layer_spectrum(weights: np.ndarray) -> np.ndarray:
    """Singular values of ``weights`` in descending order."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D matrix")
    return np.linalg.svd(weights, compute_uv=False)


def effective_rank(weights: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose singular values cover ``energy`` of total energy."""
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    s = layer_spectrum(weights)
    total = float(s.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(s) / total
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    return int(min(k, s.size))


def layer_report(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    s = layer_spectrum(weights)
    s_max = float(s[0])
    s_min = float(s[-1])
    return {
        "shape": (int(weights.shape[0]), int(weights.shape[1])),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "spectral_norm": s_max,
        # Numerically singular: s_min below machine-eps scale => inf
        "condition_number": (
            s_max / s_min
            if s_min > s_max * np.finfo(np.float64).eps * max(weights.shape)
            else float("inf")
        ),
        "effective_rank": effective_rank(weights),
    }


def analyze_model(layers: dict) -> dict:
    """Report every layer of a model dict {name: 2-D matrix}."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return {name: layer_report(w) for name, w in layers.items()}


def find_bottlenecks(reports: dict, top_k: int = 1) -> list:
    """Layer names with the smallest effective rank, worst first."""
    if not reports:
        return []
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = sorted(reports.items(), key=lambda kv: (kv[1]["effective_rank"], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def spectrum_distance(baseline: np.ndarray, candidate: np.ndarray, bins: int = 16) -> float:
    """Distance between the singular-value spectra of two matrices.

    Both spectra are log-transformed, normalized to unit sum, and compared
    on ``bins`` evenly spaced quantile points. Scale differences between
    matrices are ignored; only the shape of the decay is compared.

    Returns a float >= 0. 0.0 means identical spectral shape.
    """
    s1 = layer_spectrum(baseline)
    s2 = layer_spectrum(candidate)
    if s1.size == 0 or s2.size == 0:
        raise ValueError("inputs must be non-empty 2-D matrices")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    def _profile(s: np.ndarray) -> np.ndarray:
        # Normalize by max BEFORE log so scaling the matrix leaves the
        # profile unchanged: log(c*s / c*s_max) == log(s / s_max).
        s_norm = s / s.max()
        log_s = np.log1p(s_norm)
        quantiles = np.linspace(0.0, 1.0, bins)
        profile = np.quantile(log_s, quantiles)
        total = float(np.abs(profile).sum())
        return profile / total if total > 0 else profile

    return float(np.abs(_profile(s1) - _profile(s2)).sum())


def shift_report(baseline_layers: dict, candidate_layers: dict, bins: int = 16) -> dict:
    """Layer-by-layer spectral shift between two representation sets.

    Args:
        baseline_layers: dict {layer_name: 2-D matrix} from the baseline.
        candidate_layers: dict {layer_name: 2-D matrix} from the candidate.
        bins: quantile bins for spectrum_distance.

    Returns dict {layer_name: distance} for layers present in both, plus
    a "most_shifted" key naming the layer with the largest distance.
    """
    if not baseline_layers or not candidate_layers:
        raise ValueError("both layer dicts must be non-empty")
    common = sorted(set(baseline_layers) & set(candidate_layers))
    if not common:
        raise ValueError("no common layer names between the two sets")
    distances = {
        name: spectrum_distance(baseline_layers[name], candidate_layers[name], bins=bins)
        for name in common
    }
    distances["most_shifted"] = max(common, key=lambda n: distances[n])
    return distances