"""Sampling utilities for the stopping distance simulator.

Dependencies: numpy, scipy
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import scipy.stats as stats
import math

# ------------------ Speed ------------------

def speed_params(v_disp: float) -> Tuple[float, float, float]:
    """Return (min, mode, max) parameters for the real speed."""
    delta = min(4 + 0.05 * v_disp, 8)
    return v_disp - delta, v_disp - delta / 2, v_disp


def sample_speed(v_disp: float, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    a, c, b = speed_params(v_disp)
    return rng.triangular(a, c, b, n)


def speed_pdf(x: np.ndarray, v_disp: float) -> np.ndarray:
    a, c, b = speed_params(v_disp)
    return np.where(
        (x >= a) & (x <= b),
        np.where(x < c, 2 * (x - a) / ((b - a) * (c - a)), 2 * (b - x) / ((b - a) * (b - c))),
        0,
    )

# ------------------ Reaction time ------------------

PROFILE_MED = {"Alerte": 0.9, "Standard": 1.5, "Fatigué": 2.0, "Senior": 2.0}
K_WEIB = 2.2


def weib_scale(median: float) -> float:
    return median / (math.log(2) ** (1 / K_WEIB))


def sample_tr(profile: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample reaction times from a truncated Weibull distribution."""
    lam = weib_scale(PROFILE_MED[profile])
    rng = rng or np.random.default_rng()
    x = stats.weibull_min.rvs(K_WEIB, scale=lam, size=n, random_state=rng)
    mask = (x < 0.3) | (x > 3)
    while mask.any():
        x[mask] = stats.weibull_min.rvs(K_WEIB, scale=lam, size=mask.sum(), random_state=rng)
        mask = (x < 0.3) | (x > 3)
    return x


def tr_pdf(x: np.ndarray, profile: str) -> np.ndarray:
    lam = weib_scale(PROFILE_MED[profile])
    pdf = stats.weibull_min.pdf(x, K_WEIB, scale=lam)
    pdf[(x < 0.3) | (x > 3)] = 0
    norm = stats.weibull_min.cdf(3, K_WEIB, scale=lam) - stats.weibull_min.cdf(0.3, K_WEIB, scale=lam)
    pdf /= norm
    return pdf

# ------------------ Tyre adhesion ------------------

SURFACE_μ = {
    "sec": {"neuf": 0.85, "mi-usure": 0.80, "usé": 0.75},
    "mouillé": {"neuf": 0.55, "mi-usure": 0.47, "usé": 0.40},
    "neige": {"neuf": 0.25, "mi-usure": 0.25, "usé": 0.25},
    "glace": {"neuf": 0.10, "mi-usure": 0.10, "usé": 0.10},
}
A_B, B_B = 2, 3


def base_mu(surface: str, tyre: str) -> float:
    mu = SURFACE_μ[surface][tyre]
    return float(np.clip(mu, 0.2, 0.9))


def mu_bounds(mu: float) -> Tuple[float, float]:
    return max(0.2, mu - 0.15), min(0.9, mu + 0.15)


def sample_mu(surface: str, tyre: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    mu0 = base_mu(surface, tyre)
    mu_min, mu_max = mu_bounds(mu0)
    rng = rng or np.random.default_rng()
    return mu_min + (mu_max - mu_min) * rng.beta(A_B, B_B, size=n)


def mu_pdf(x: np.ndarray, surface: str, tyre: str) -> np.ndarray:
    mu0 = base_mu(surface, tyre)
    mu_min, mu_max = mu_bounds(mu0)
    pdf = stats.beta.pdf((x - mu_min) / (mu_max - mu_min), A_B, B_B) / (mu_max - mu_min)
    pdf[(x < mu_min) | (x > mu_max)] = 0
    return pdf

# ------------------ Slope ------------------

SLOPE = {"Plat": 0, "Montée 2°": 2, "Montée 4°": 4, "Descente 2°": -2, "Descente 4°": -4}


def sample_theta(cat: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    mu = SLOPE[cat]
    a, b = (-1) / 0.5, 1 / 0.5
    rng = rng or np.random.default_rng()
    return stats.truncnorm.rvs(a, b, loc=mu, scale=0.5, size=n, random_state=rng)


def theta_pdf(x: np.ndarray, cat: str) -> np.ndarray:
    mu = SLOPE[cat]
    a, b = (-1) / 0.5, 1 / 0.5
    pdf = stats.truncnorm.pdf(x, a, b, loc=mu, scale=0.5)
    pdf[(x < mu - 1) | (x > mu + 1)] = 0
    return pdf
