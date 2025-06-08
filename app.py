# app.py — Simulateur de distance d’arrêt
"""Simulation interactive de la distance d'arrêt d'un véhicule.

Dépendances principales : Streamlit pour l'interface, NumPy et SciPy pour
les calculs scientifiques et Plotly pour la visualisation.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
import textwrap
import math
import time

st.set_page_config(page_title="Simulateur – Distance d’arrêt",
                   page_icon="🚗", layout="wide")
st.title("Simulateur de distance d'arrêt")
G = 9.81  # gravité (m·s-2)
RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class Params:
    """Regroupe les paramètres de la simulation."""

    speed: int
    profile: str
    surface: str
    tyre: str
    slope: str
    conf: float
    child_d: float

# ==============================================================
# 1. Lois de probabilité
# ==============================================================

# ---- Vitesse réelle (triangulaire, compteur sur-estimant) ----
def speed_params(v_disp: float) -> Tuple[float, float, float]:
    """Bornes et mode de la distribution triangulaire de vitesse."""
    Δ = min(4 + 0.05 * v_disp, 8)           # tolérance UN-R39
    return v_disp - Δ, v_disp - Δ / 2, v_disp

def sample_speed(v_disp: float, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Tire ``n`` vitesses réelles."""
    rng = rng or RNG
    a, c, b = speed_params(v_disp)
    return rng.triangular(a, c, b, n)

def speed_pdf(x: np.ndarray, v_disp: float) -> np.ndarray:
    """Densité de la vitesse réelle."""
    a, c, b = speed_params(v_disp)
    return np.where(
        (x >= a) & (x <= b),
        np.where(x < c,
                 2*(x-a)/((b-a)*(c-a)),
                 2*(b-x)/((b-a)*(b-c))),
        0)

# ---- Temps de réaction (Weibull tronqué) ----
PROFILE_MED = {
    "Alerte": 0.9,
    "Standard": 1.5,
    "Fatigué": 2.0,
    "Senior": 2.0,
}
K_WEIB = 2.2

def weib_scale(med: float) -> float:
    """Paramètre d'échelle λ d'une Weibull pour une médiane donnée."""
    return med / (math.log(2) ** (1 / K_WEIB))

def sample_tr(profile: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Échantillonne des temps de réaction tronqués."""
    lam = weib_scale(PROFILE_MED[profile])
    rng = rng or RNG
    x = stats.weibull_min.rvs(K_WEIB, scale=lam, size=n, random_state=rng)
    mask = (x < 0.3) | (x > 3)
    while mask.any():
        x[mask] = stats.weibull_min.rvs(
            K_WEIB, scale=lam, size=mask.sum(), random_state=rng
        )
        mask = (x < 0.3) | (x > 3)
    return x

def tr_pdf(x: np.ndarray, profile: str) -> np.ndarray:
    """Densité normalisée du temps de réaction tronqué."""
    lam = weib_scale(PROFILE_MED[profile])
    prob = stats.weibull_min.cdf(3, K_WEIB, scale=lam) - stats.weibull_min.cdf(
        0.3, K_WEIB, scale=lam
    )
    pdf = stats.weibull_min.pdf(x, K_WEIB, scale=lam) / prob
    pdf[(x < 0.3) | (x > 3)] = 0
    return pdf

# ---- Adhérence μ (Bêta bornée) ----
SURFACE_μ = {
    "sec":     {"neuf": .85, "mi-usure": .80, "usé": .75},
    "mouillé": {"neuf": .55, "mi-usure": .47, "usé": .40},
    "neige":   {"neuf": .25, "mi-usure": .25, "usé": .25},
    "glace":   {"neuf": .10, "mi-usure": .10, "usé": .10},
}
A_B, B_B = 2, 3

def base_mu(surface: str, tyre: str) -> float:
    """Valeur nominale d'adhérence suivant surface et état des pneus."""
    μ = SURFACE_μ[surface][tyre]
    return np.clip(μ, 0.2, 0.9)

def mu_bounds(μ: float) -> Tuple[float, float]:
    """Bornes minimales et maximales de μ pour la simulation."""
    return max(0.2, μ - 0.15), min(0.9, μ + 0.15)

def sample_mu(surface: str, tyre: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Échantillonne le coefficient d'adhérence."""
    μ0 = base_mu(surface, tyre)
    μ_min, μ_max = mu_bounds(μ0)
    rng = rng or RNG
    return μ_min + (μ_max - μ_min) * rng.beta(A_B, B_B, size=n)

def mu_pdf(x: np.ndarray, surface: str, tyre: str) -> np.ndarray:
    """Densité du coefficient d'adhérence."""
    μ0 = base_mu(surface, tyre)
    μ_min, μ_max = mu_bounds(μ0)
    pdf = stats.beta.pdf((x - μ_min) / (μ_max - μ_min), A_B, B_B) / (μ_max - μ_min)
    pdf[(x < μ_min) | (x > μ_max)] = 0
    return pdf

# ---- Pente θ (normale tronquée ±1° autour du centre) ----
SLOPE = {
    "Plat": 0,
    "Montée 2°": 2,
    "Montée 4°": 4,
    "Descente 2°": -2,
    "Descente 4°": -4,
}

def sample_theta(cat: str, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Échantillonne l'angle de pente."""
    μ = SLOPE[cat]
    a, b = (-1) / 0.5, 1 / 0.5
    rng = rng or RNG
    return stats.truncnorm.rvs(a, b, loc=μ, scale=0.5, size=n, random_state=rng)

def theta_pdf(x: np.ndarray, cat: str) -> np.ndarray:
    """Densité de la pente tronquée."""
    μ = SLOPE[cat]; a, b = (-1) / 0.5, 1 / 0.5
    pdf = stats.truncnorm.pdf(x, a, b, loc=μ, scale=0.5)
    pdf[(x < μ - 1) | (x > μ + 1)] = 0
    return pdf

# ==============================================================
# 2. Modèle physique (distance d’arrêt)
# ==============================================================

def stopping_distance(
    v_kmh: np.ndarray, t_r: np.ndarray, μ: np.ndarray, θ_deg: np.ndarray
) -> np.ndarray:
    """Calcule la distance d'arrêt pour chaque tirage."""
    v_ms = v_kmh / 3.6
    θ = np.radians(θ_deg)
    denom = 2 * G * (μ * np.cos(θ) + np.sin(θ))
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = v_ms * t_r + (v_ms ** 2) / denom
    dist[denom <= 0] = np.inf
    return dist

# ==============================================================
# 3. Monte-Carlo adaptatif
# ==============================================================

@st.cache_data(show_spinner=False, hash_funcs={Callable: lambda _: None})
def run_mc(
    p: Params,
    batch: int = 50_000,
    max_iter: int = 20,
    _stop_flag: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Monte-Carlo adaptatif produisant les distances d'arrêt."""

    z = stats.norm.ppf(0.5 + p.conf / 2)
    rel_tol = 1 - p.conf
    dist_chunks = []
    progress = st.progress(0)

    for i in range(max_iter):
        if _stop_flag and _stop_flag():
            raise RuntimeError("Simulation interrompue")

        v = sample_speed(p.speed, batch)
        t = sample_tr(p.profile, batch)
        μ = sample_mu(p.surface, p.tyre, batch)
        θ = sample_theta(p.slope, batch)

        ok = μ * np.cos(np.radians(θ)) + np.sin(np.radians(θ)) > 0
        while not ok.all():
            idx = np.where(~ok)[0]
            v[idx] = sample_speed(p.speed, len(idx))
            t[idx] = sample_tr(p.profile, len(idx))
            μ[idx] = sample_mu(p.surface, p.tyre, len(idx))
            θ[idx] = sample_theta(p.slope, len(idx))
            ok = μ * np.cos(np.radians(θ)) + np.sin(np.radians(θ)) > 0

        assert ok.all(), "Invalid combinaison donnant denom <= 0"

        dist_chunks.append(stopping_distance(v, t, μ, θ))
        dist_all = np.concatenate(dist_chunks)
        sem = np.std(dist_all, ddof=1) / np.sqrt(len(dist_all))
        progress.progress(int((i + 1) / max_iter * 100))
        if z * sem / dist_all.mean() < rel_tol:
            break
    progress.empty()
    return np.concatenate(dist_chunks)

# ==============================================================
# 4. Interface Streamlit
# ==============================================================

st.sidebar.header("Paramètres")
advanced = st.sidebar.toggle("Mode avancé")

if advanced:
    with st.sidebar.expander("Paramètres avancés", expanded=True):
        speed = st.slider(
            "Vitesse compteur (km/h)",
            30,
            130,
            90,
            step=5,
            help="Vitesse affichée au compteur",
        )
        profile = st.radio(
            "Profil conducteur",
            list(PROFILE_MED),
            1,
            help="Temps de réaction médian selon le conducteur",
        )
        surface = st.select_slider(
            "Chaussée 🚧",
            options=list(SURFACE_μ),
            help="État de la chaussée",
        )
        tyre = st.select_slider(
            "Pneus 🔄",
            options=list(SURFACE_μ["sec"].keys()),
            help="Usure des pneumatiques",
        )
        slope = st.radio("Pente", list(SLOPE), 0, help="Inclinaison de la route")
        conf = st.slider("Confiance (%)", 0, 100, 95, help="Niveau de confiance") / 100
else:
    PRESETS = {
        "Ville – chaussée sèche": {
            "desc": "30 km/h, conducteur standard, pneus neufs",
            "speed": 30,
            "profile": "Standard",
            "surface": "sec",
            "tyre": "neuf",
            "slope": "Plat",
        },
        "Ville – chaussée mouillée": {
            "desc": "30 km/h, conducteur fatigué, pneus mi-usure",
            "speed": 30,
            "profile": "Fatigué",
            "surface": "mouillé",
            "tyre": "mi-usure",
            "slope": "Plat",
        },
        "Route – chaussée sèche": {
            "desc": "80 km/h, conducteur standard, pneus mi-usure",
            "speed": 80,
            "profile": "Standard",
            "surface": "sec",
            "tyre": "mi-usure",
            "slope": "Plat",
        },
        "Route – chaussée mouillée": {
            "desc": "80 km/h, conducteur senior, pneus usés",
            "speed": 80,
            "profile": "Senior",
            "surface": "mouillé",
            "tyre": "usé",
            "slope": "Plat",
        },
        "Autoroute – chaussée sèche": {
            "desc": "130 km/h, conducteur alerte, pneus neufs",
            "speed": 130,
            "profile": "Alerte",
            "surface": "sec",
            "tyre": "neuf",
            "slope": "Plat",
        },
        "Autoroute – chaussée mouillée": {
            "desc": "110 km/h, conducteur standard, pneus mi-usure",
            "speed": 110,
            "profile": "Standard",
            "surface": "mouillé",
            "tyre": "mi-usure",
            "slope": "Plat",
        },
    }
    options = list(PRESETS)
    preset_name = st.sidebar.radio(
        "Préréglage", options, captions=[PRESETS[o]["desc"] for o in options]
    )
    pr = PRESETS[preset_name]
    speed = pr["speed"]
    profile = pr["profile"]
    surface = pr["surface"]
    tyre = pr["tyre"]
    slope = pr["slope"]
    conf = 0.95

child_d = st.sidebar.slider(
    "Distance de l'enfant (m)", 5.0, 100.0, 25.0, step=0.1, help="Position de l'enfant"
)
run_sim = st.sidebar.button("Lancer la simulation")
stop_sim = st.sidebar.button("\u23F9\ufe0f Stop")

params = Params(
    speed=speed,
    profile=profile,
    surface=surface,
    tyre=tyre,
    slope=slope,
    conf=conf,
    child_d=child_d,
)

tab_res, tab_stats, tab_var, tab_about = st.tabs([
    "📊 Résultats",
    "📋 Statistiques",
    "🔎 Variables",
    "ℹ️ À propos",
])

# --------------------------------------------------------------
# 5. Exécution / affichage
# --------------------------------------------------------------
if "dist" not in st.session_state:
    st.session_state["dist"] = None
    st.session_state["params"] = None

if run_sim:
    st.session_state["stop"] = False
    t0 = time.time()
    try:
        with st.spinner("Simulation en cours..."):
            dist = run_mc(
                params,
                _stop_flag=lambda: st.session_state.get("stop", False),
            )
    except RuntimeError as exc:
        st.error(str(exc))
        dist = None
    dt = time.time() - t0
    st.session_state["dist"] = dist
    st.session_state["params"] = params
elif stop_sim:
    st.session_state["stop"] = True
elif (
    st.session_state["dist"] is not None
    and st.session_state["params"] == params
):
    dist = st.session_state["dist"]
    dt = 0
else:
    dist = None
if dist is not None:
    mean = dist.mean()
    std = dist.std(ddof=1)
    p_val = int(params.conf * 100)
    p_quant = np.percentile(dist, p_val)
    p_coll = (dist >= child_d).mean()
    z = stats.norm.ppf(0.5 + params.conf / 2)
    ci = z * std

    # -------- Graphiques et KPIs --------------------------------------
    with tab_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance moyenne (m)", f"{mean:.1f}")
        c2.metric("Écart type (m)", f"{std:.1f}")
        c3.metric(f"Distance P{p_val} (m)", f"{p_quant:.1f}")
        c4.metric("Probabilité de collision", f"{p_coll*100:.1f} %")

        fig_hist = px.histogram(
            dist,
            nbins=60,
            labels={"value": "Distance d'arrêt (m)", "percent": "Fréquence (%)"},
            template="plotly_white",
            title="Distribution simulée",
            histnorm="percent",
        )
        fig_hist.update_yaxes(title="Fréquence (%)")
        st.plotly_chart(fig_hist, use_container_width=True)

        sorted_dist = np.sort(dist)
        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        fig_cdf = px.area(
            x=sorted_dist,
            y=cdf * 100,
            labels={"x": "Distance d'arrêt (m)", "y": "Probabilité (%)"},
            template="plotly_white",
        )
        fig_cdf.update_yaxes(range=[0, 100])
        fig_cdf.add_vline(
            x=child_d,
            line_dash="dash",
            line_width=2,
            annotation_text="Position de l’enfant",
            annotation_position="top",
        )
        fig_cdf.add_shape(
            type="rect",
            x0=child_d,
            x1=sorted_dist.max(),
            y0=0,
            y1=1,
            fillcolor="red",
            opacity=0.2,
            line_width=0,
        )
        fig_cdf.update_layout(title="Probabilité cumulée de collision")
        st.plotly_chart(fig_cdf, use_container_width=True)

        st.caption(
            f"{format(len(dist), ',').replace(',', '\u202f')} tirages – {dt:.2f}s"
        )

    # -------- Statistiques --------------------------------------------
    with tab_stats:
        st.subheader("Statistiques")
        st.write(
            f"La distance d'arrêt moyenne est **{mean:.1f} ± {ci:.1f} m** "+
            f"(niveau de confiance {params.conf*100:.0f} %)."
        )
        q25, q50, q75 = np.percentile(dist, [25, 50, 75])
        st.markdown(
            f"Minimum : {dist.min():.1f} m  \n"
            f"1er quartile : {q25:.1f} m  \n"
            f"Médiane : {q50:.1f} m  \n"
            f"3e quartile : {q75:.1f} m  \n"
            f"Maximum : {dist.max():.1f} m"
        )
        fig_box = px.box(
            dist,
            points=False,
            labels={"value": "Distance d'arrêt (m)"},
            template="plotly_white",
            title="Boîte à moustaches"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # -------- Distributions internes ----------------------------------
    with tab_var:
        st.subheader("Distributions internes")
        st.markdown(
            "- **Vitesse réelle :** loi triangulaire\n"
            "- **Temps de réaction :** loi de Weibull tronquée\n"
            "- **Adhérence μ :** loi bêta bornée\n"
            "- **Pente θ :** loi normale tronquée"
        )
        rng = RNG

        with st.expander("Vitesse réelle"):
            xs = np.linspace(speed_params(speed)[0], speed, 300)
            data = sample_speed(speed, 10_000, rng)
            fig = px.histogram(
                data,
                nbins=40,
                histnorm="probability density",
                opacity=0.6,
                template="plotly_white",
            )
            fig.add_scatter(x=xs, y=speed_pdf(xs, speed))
            fig.update_layout(title="Vitesse réelle (km/h)")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Temps de réaction"):
            xs = np.linspace(0.3, 3, 300)
            data = sample_tr(profile, 10_000, rng)
            fig = px.histogram(
                data,
                nbins=40,
                histnorm="probability density",
                opacity=0.6,
                template="plotly_white",
            )
            fig.add_scatter(x=xs, y=tr_pdf(xs, profile))
            fig.update_layout(title="Temps de réaction (s)")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Adhérence μ"):
            μ_min, μ_max = mu_bounds(base_mu(surface, tyre))
            xs = np.linspace(μ_min, μ_max, 300)
            data = sample_mu(surface, tyre, 10_000, rng)
            fig = px.histogram(
                data,
                nbins=40,
                histnorm="probability density",
                opacity=0.6,
                template="plotly_white",
            )
            fig.add_scatter(x=xs, y=mu_pdf(xs, surface, tyre))
            fig.update_layout(title="Coefficient d'adhérence μ")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Pente θ"):
            μθ = SLOPE[slope]
            xs = np.linspace(μθ - 1, μθ + 1, 300)
            data = sample_theta(slope, 10_000, rng)
            fig = px.histogram(
                data,
                nbins=40,
                histnorm="probability density",
                opacity=0.6,
                template="plotly_white",
            )
            fig.add_scatter(x=xs, y=theta_pdf(xs, slope))
            fig.update_layout(title="Angle de pente θ (°)")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
else:
    tab_res.info("Aucun résultat pour l'instant.")
    tab_stats.info("Aucun résultat pour l'instant.")
    tab_var.markdown("_Les distributions apparaîtront après simulation._")

# ------------------ À propos ----------------------------------
with tab_about:
    st.markdown("### Vos paramètres actuels")
    saved = st.session_state.get("params")
    if saved:
        mu_base = base_mu(saved.surface, saved.tyre)
        tr_nom = PROFILE_MED[saved.profile]
        st.markdown(
            textwrap.dedent(
                f"""
                • **Vitesse compteur :** {saved.speed} km/h<br>
                • **Profil conducteur :** {saved.profile}  – temps de réaction médian ≈ {tr_nom:.1f} s<br>
                • **Chaussée :** {saved.surface}<br>
                • **Pneus :** {saved.tyre}<br>
                • **Adhérence nominale μ :** {mu_base:.2f} (plage simulée ±0,15)<br>
                • **Pente :** {SLOPE[saved.slope]:+} ° ({saved.slope})<br>
                • **Confiance MC :** {saved.conf*100:.0f} %<br>
                • **Distance enfant :** {saved.child_d} m
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info("Aucune simulation pour l'instant.")
