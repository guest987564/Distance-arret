"""Standalone Streamlit app for stopping distance simulation.

This single file contains the sampling utilities, physical model,
Monte-Carlo engine and user interface."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
import plotly.express as px
import math

from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
import scipy.stats as stats

rng = np.random.default_rng()

# ------------------ Speed ------------------

def speed_params(v_disp: float) -> Tuple[float, float, float]:
    """Return (min, mode, max) parameters for the real speed."""
    delta = min(4 + 0.05 * v_disp, 8)
    return v_disp - delta, v_disp - delta / 2, v_disp


def sample_speed(v_disp: float, n: int, rng_: Optional[np.random.Generator] = None) -> np.ndarray:
    r = rng_ or rng
    a, c, b = speed_params(v_disp)
    return r.triangular(a, c, b, n)


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


def sample_tr(profile: str, n: int, rng_: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample reaction times from a truncated Weibull distribution."""
    lam = weib_scale(PROFILE_MED[profile])
    r = rng_ or rng
    x = stats.weibull_min.rvs(K_WEIB, scale=lam, size=n, random_state=r)
    mask = (x < 0.3) | (x > 3)
    while mask.any():
        x[mask] = stats.weibull_min.rvs(K_WEIB, scale=lam, size=mask.sum(), random_state=r)
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


def sample_mu(surface: str, tyre: str, n: int, rng_: Optional[np.random.Generator] = None) -> np.ndarray:
    mu0 = base_mu(surface, tyre)
    mu_min, mu_max = mu_bounds(mu0)
    r = rng_ or rng
    return mu_min + (mu_max - mu_min) * r.beta(A_B, B_B, size=n)


def mu_pdf(x: np.ndarray, surface: str, tyre: str) -> np.ndarray:
    mu0 = base_mu(surface, tyre)
    mu_min, mu_max = mu_bounds(mu0)
    pdf = stats.beta.pdf((x - mu_min) / (mu_max - mu_min), A_B, B_B) / (mu_max - mu_min)
    pdf[(x < mu_min) | (x > mu_max)] = 0
    return pdf

# ------------------ Slope ------------------

SLOPE = {"Plat": 0, "Montée 2°": 2, "Montée 4°": 4, "Descente 2°": -2, "Descente 4°": -4}


def sample_theta(cat: str, n: int, rng_: Optional[np.random.Generator] = None) -> np.ndarray:
    mu = SLOPE[cat]
    a, b = (-1) / 0.5, 1 / 0.5
    r = rng_ or rng
    return stats.truncnorm.rvs(a, b, loc=mu, scale=0.5, size=n, random_state=r)


def theta_pdf(x: np.ndarray, cat: str) -> np.ndarray:
    mu = SLOPE[cat]
    a, b = (-1) / 0.5, 1 / 0.5
    pdf = stats.truncnorm.pdf(x, a, b, loc=mu, scale=0.5)
    pdf[(x < mu - 1) | (x > mu + 1)] = 0
    return pdf

# ------------------ Physical model ------------------

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
    """Return stopping distance in meters."""
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
    local_rng = np.random.default_rng(p.seed) if p.seed is not None else rng
    z = stats.norm.ppf(0.5 + p.conf / 2)
    rel_tol = 1 - p.conf
    chunks: List[np.ndarray] = []

    for i in range(max_iter):
        if st.session_state.get("stop"):
            raise RuntimeError("Simulation cancelled")

        v = sample_speed(p.speed, batch, local_rng)
        t = sample_tr(p.profile, batch, local_rng)
        mu = sample_mu(p.surface, p.tyre, batch, local_rng)
        theta = sample_theta(p.slope, batch, local_rng)

        denom_ok = mu * np.cos(np.radians(theta)) + np.sin(np.radians(theta)) > 0
        while not denom_ok.all():
            idx = np.where(~denom_ok)[0]
            v[idx] = sample_speed(p.speed, len(idx), local_rng)
            t[idx] = sample_tr(p.profile, len(idx), local_rng)
            mu[idx] = sample_mu(p.surface, p.tyre, len(idx), local_rng)
            theta[idx] = sample_theta(p.slope, len(idx), local_rng)
            denom_ok = mu * np.cos(np.radians(theta)) + np.sin(np.radians(theta)) > 0

        chunks.append(stopping_distance(v, t, mu, theta))
        dist = np.concatenate(chunks)
        sem = np.std(dist, ddof=1) / np.sqrt(len(dist))
        if progress_callback:
            progress_callback(int((i + 1) / max_iter * 100))
        if z * sem / dist.mean() < rel_tol:
            break

    return dist

st.set_page_config(page_title="Simulateur – Distance d'arrêt", page_icon="🚗", layout="wide")
st.title("Simulateur de distance d'arrêt")

st.sidebar.header("Configuration")

with st.sidebar.expander("Paramètres avancés"):
    speed = st.slider("Vitesse compteur (km/h)", 30, 130, 90, step=5, help="Vitesse affichée au compteur")
    profile = st.selectbox("Profil conducteur", list(PROFILE_MED), index=1, help="Temps de réaction médian")
    surface = st.select_slider("Chaussée 🚧", options=list(SURFACE_μ), value="sec", help="État de la chaussée")
    tyre = st.select_slider("Pneus 🔄", options=list(SURFACE_μ["sec"].keys()), value="neuf", help="Usure des pneus")
    slope = st.select_slider("Pente", options=list(SLOPE), value="Plat", help="Inclinaison de la route")
    conf = st.slider("Confiance (%)", 50, 99, 95, help="Intervalle de confiance de la MC") / 100
    seed_opt = st.number_input("Seed aléatoire", value=0, step=1, help="Optionnel pour reproduire")
    use_seed = st.checkbox("Utiliser le seed", value=False)

child_d = st.sidebar.slider("Distance de l'enfant (m)", 5.0, 100.0, 25.0, step=0.1)
run_sim = st.sidebar.button("Lancer la simulation")
stop_sim = st.sidebar.button("⏹️ Stop")

params = SimParams(
    speed=speed,
    profile=profile,
    surface=surface,
    tyre=tyre,
    slope=slope,
    conf=conf,
    child_d=child_d,
    seed=int(seed_opt) if use_seed else None,
)

if "dist" not in st.session_state:
    st.session_state["dist"] = None
    st.session_state["params"] = None

if stop_sim:
    st.session_state["stop"] = True

if run_sim:
    st.session_state["stop"] = False
    progress = st.progress(0)
    try:
        t0 = time.time()
        dist = run_mc(params, progress_callback=progress.progress)
        dt = time.time() - t0
        st.session_state["dist"] = dist
        st.session_state["params"] = params
    except RuntimeError:
        st.warning("Simulation annulée.")
        dist = None
        dt = 0
    progress.empty()
elif st.session_state.get("dist") is not None and st.session_state.get("params") == params:
    dist = st.session_state["dist"]
    dt = 0
else:
    dist = None

if dist is not None:
    mean = float(np.mean(dist))
    p95 = float(np.percentile(dist, 95))
    p_coll = float((dist >= child_d).mean())
    st.write(f"Simulation terminée en {dt:.2f}s")

    tab_graph, tab_var, tab_about = st.tabs(["📈 Graphiques", "🔎 Variables", "ℹ️ À propos"])

    with tab_graph:
        c1, c2, c3 = st.columns(3)
        c1.metric("Distance moyenne (m)", f"{mean:.1f}")
        c2.metric("Distance P95 (m)", f"{p95:.1f}")
        c3.metric("Probabilité de collision", f"{p_coll*100:.1f} %")

        fig_hist = px.histogram(dist, nbins=60, labels={"value": "Distance d'arrêt (m)"}, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

        sorted_dist = np.sort(dist)
        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        fig_cdf = px.line(x=sorted_dist, y=cdf, labels={"x": "Distance d'arrêt (m)", "y": "Probabilité"}, template="plotly_white")
        fig_cdf.add_vline(x=child_d, line_dash="dash", line_width=2, annotation_text="Position de l’enfant", annotation_position="top")
        cdf_at_child = float(np.interp(child_d, sorted_dist, cdf))
        fig_cdf.add_shape(type="rect", x0=child_d, x1=sorted_dist[-1], y0=cdf_at_child, y1=1, fillcolor="rgba(255,0,0,0.2)", line_width=0)
        st.plotly_chart(fig_cdf, use_container_width=True)

        st.caption(f"{format(len(dist), ',').replace(',', ' ')} tirages – {dt:.2f}s")

    with tab_var:
        st.subheader("Distributions internes")
        with st.expander("Vitesse réelle"):
            xs = np.linspace(speed_params(speed)[0], speed, 300)
            data = sample_speed(speed, 10_000, rng)
            fig = px.histogram(data, nbins=40, histnorm="probability density", template="plotly_white")
            fig.add_scatter(x=xs, y=speed_pdf(xs, speed), name="PDF")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Temps de réaction"):
            xs = np.linspace(0.3, 3, 300)
            data = sample_tr(profile, 10_000, rng)
            fig = px.histogram(data, nbins=40, histnorm="probability density", template="plotly_white")
            fig.add_scatter(x=xs, y=tr_pdf(xs, profile), name="PDF")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Adhérence μ"):
            mu_min, mu_max = mu_bounds(base_mu(surface, tyre))
            xs = np.linspace(mu_min, mu_max, 300)
            data = sample_mu(surface, tyre, 10_000, rng)
            fig = px.histogram(data, nbins=40, histnorm="probability density", template="plotly_white")
            fig.add_scatter(x=xs, y=mu_pdf(xs, surface, tyre), name="PDF")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Pente θ"):
            mu_theta = SLOPE[slope]
            xs = np.linspace(mu_theta - 1, mu_theta + 1, 300)
            data = sample_theta(slope, 10_000, rng)
            fig = px.histogram(data, nbins=40, histnorm="probability density", template="plotly_white")
            fig.add_scatter(x=xs, y=theta_pdf(xs, slope), name="PDF")
            st.plotly_chart(fig, use_container_width=True)

    with tab_about:
        st.markdown("### Vos paramètres actuels")
        st.json(params.__dict__)
else:
    st.info("Aucune simulation pour l'instant.")
