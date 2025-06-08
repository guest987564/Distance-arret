"""Core Monte-Carlo model for stopping distance simulation.

Dependencies: numpy, scipy, streamlit
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List
import numpy as np
import scipy.stats as stats
import streamlit as st

from sampling import (
    sample_speed,
    sample_tr,
    sample_mu,
    sample_theta,
)

G = 9.81  # gravité (m·s-2)


@dataclass(frozen=True)
class SimParams:
    """Parameters controlling a Monte-Carlo run."""

    speed: float
    profile: str
    surface: str
    tyre: str
    slope: str
    conf: float
    child_d: float
    seed: Optional[int] = None


def stopping_distance(v_kmh: np.ndarray, t_r: np.ndarray, mu: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """Return stopping distance in meters.

    Raises
    ------
    ValueError
        If a non-positive denominator occurs.
    """
    v_ms = v_kmh / 3.6
    theta = np.radians(theta_deg)
    denom = 2 * G * (mu * np.cos(theta) + np.sin(theta))
    if np.any(denom <= 0):
        raise ValueError("Invalid parameter combination leading to denom <= 0")
    return v_ms * t_r + (v_ms ** 2) / denom


@st.cache_data(show_spinner=False)
def run_mc(
    p: SimParams,
    batch: int = 50_000,
    max_iter: int = 20,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Run the adaptive Monte Carlo algorithm."""
    rng = np.random.default_rng(p.seed)
    z = stats.norm.ppf(0.5 + p.conf / 2)
    rel_tol = 1 - p.conf
    chunks: List[np.ndarray] = []

    for i in range(max_iter):
        if st.session_state.get("stop"):
            raise RuntimeError("Simulation cancelled")

        v = sample_speed(p.speed, batch, rng)
        t = sample_tr(p.profile, batch, rng)
        mu = sample_mu(p.surface, p.tyre, batch, rng)
        theta = sample_theta(p.slope, batch, rng)

        denom_ok = mu * np.cos(np.radians(theta)) + np.sin(np.radians(theta)) > 0
        while not denom_ok.all():
            idx = np.where(~denom_ok)[0]
            v[idx] = sample_speed(p.speed, len(idx), rng)
            t[idx] = sample_tr(p.profile, len(idx), rng)
            mu[idx] = sample_mu(p.surface, p.tyre, len(idx), rng)
            theta[idx] = sample_theta(p.slope, len(idx), rng)
            denom_ok = mu * np.cos(np.radians(theta)) + np.sin(np.radians(theta)) > 0

        chunks.append(stopping_distance(v, t, mu, theta))
        dist = np.concatenate(chunks)
        sem = np.std(dist, ddof=1) / np.sqrt(len(dist))
        if progress_callback:
            progress_callback(int((i + 1) / max_iter * 100))
        if z * sem / dist.mean() < rel_tol:
            break

    return dist
