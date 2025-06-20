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
import plotly.io as pio
import scipy.stats as stats
import streamlit as st
import textwrap
import math
import time
import qrcode
import matplotlib.pyplot

st.set_page_config(
    page_title="Simulateur – Distance d’arrêt",
    page_icon="🚗",
    layout="wide",
)



st.title("Simulateur de distance d'arrêt")

# Le contenu d'introduction est désormais présenté dans l'onglet Accueil
G = 9.81  # gravité (m·s-2)
RNG = np.random.default_rng(42)

# ---- Paramètres communs aux graphiques ---------------------------------
def nice_ticks(max_val: float, target: int = 10) -> float:
    """Pas d'axe "agréable" pour une portée donnée."""
    if max_val <= 0:
        return 1
    step = max_val / target
    magnitude = 10 ** math.floor(math.log10(step))
    residual = step / magnitude
    if residual > 5:
        step = 10 * magnitude
    elif residual > 2:
        step = 5 * magnitude
    elif residual > 1:
        step = 2 * magnitude
    else:
        step = magnitude
    return step

# --- Thème Plotly personnalisé ------------------------------------------
COLORWAY = ["#4E79A7", "#F28E2B", "#76B7B2"]
pio.templates["custom"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["custom"].layout.colorway = COLORWAY
px.defaults.template = "custom"
px.defaults.color_discrete_sequence = COLORWAY


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
    "Très fatigué": 2.0,
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
    "sec":       {"neufs": .85, "mi-usés": .80, "usés": .75},
    "mouillé":   {"neufs": .55, "mi-usés": .47, "usés": .40},
    "neige":     {"neufs": .25, "mi-usés": .25, "usés": .25},
    "glace":     {"neufs": .10, "mi-usés": .10, "usés": .10},
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
) -> np.ndarray:
    """Monte-Carlo adaptatif produisant les distances d'arrêt."""

    z = stats.norm.ppf(0.5 + p.conf / 2)
    rel_tol = 1 - p.conf
    dist_chunks = []
    progress = st.progress(0)

    for i in range(max_iter):

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
    speed = st.slider(
        "Vitesse de la voiture (km/h)",
        30,
        130,
        90,
        step=5,
        help="Vitesse affichée au compteur",
    )
    profile = st.select_slider(
        "Profil conducteur",
        options=list(PROFILE_MED),
        value="Standard",
        help="Temps de réaction médian selon le conducteur",
    )
    surface = st.select_slider(
        "État de la route 🚧",
        options=list(SURFACE_μ),
        help="État de la route (adhérence)",
    )
    tyre = st.select_slider(
        "État des pneus 🔄",
        options=list(SURFACE_μ["sec"].keys()),
        help="Usure des pneumatiques",
    )
    slope_options = [
        "Montée 4°",
        "Montée 2°",
        "Plat",
        "Descente 2°",
        "Descente 4°",
    ]
    slope = st.select_slider(
        "Inclinaison de la route",
        options=slope_options,
        value="Plat",
        help="Inclinaison de la route",
    )
    conf = st.slider("Confiance (%)", 0, 100, 95, help="Niveau de confiance") / 100
else:
    PRESETS = {
        "Ville – chaussée sèche": {
            "desc": "30 km/h, conducteur standard, pneus neufs",
            "speed": 30,
            "profile": "Standard",
            "surface": "sec",
            "tyre": "neufs",
            "slope": "Plat",
        },
        "Ville – chaussée mouillé": {
            "desc": "30 km/h, conducteur fatigué, pneus mi-usés",
            "speed": 30,
            "profile": "Fatigué",
            "surface": "mouillé",
            "tyre": "mi-usés",
            "slope": "Plat",
        },
        "Route – chaussée sèche": {
            "desc": "80 km/h, conducteur standard, pneus mi-usés",
            "speed": 80,
            "profile": "Standard",
            "surface": "sec",
            "tyre": "mi-usés",
            "slope": "Plat",
        },
        "Route – chaussée mouillé": {
            "desc": "80 km/h, conducteur très fatigué, pneus usés",
            "speed": 80,
            "profile": "Très fatigué",
            "surface": "mouillé",
            "tyre": "usés",
            "slope": "Plat",
        },
        "Autoroute – chaussée sèche": {
            "desc": "130 km/h, conducteur alerte, pneus neufs",
            "speed": 130,
            "profile": "Alerte",
            "surface": "sec",
            "tyre": "neufs",
            "slope": "Plat",
        },
        "Autoroute – chaussée mouillé": {
            "desc": "110 km/h, conducteur standard, pneus mi-usés",
            "speed": 110,
            "profile": "Standard",
            "surface": "mouillé",
            "tyre": "mi-usés",
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
    "Distance de l'obstacle (m)",
    5.0,
    250.0,
    25.0,
    step=0.1,
    help="Position de l'obstacle",
)
run_sim = st.sidebar.button(
    "Lancer la simulation",
    type="primary",
    use_container_width=True,
)

params = Params(
    speed=speed,
    profile=profile,
    surface=surface,
    tyre=tyre,
    slope=slope,
    conf=conf,
    child_d=child_d,
)

tab_accueil, tab_res, tab_stats, tab_var, tab_about, tab_share = st.tabs([
    "🏠 Accueil",
    "📊 Tableau de bord",
    "📋 Statistiques",
    "🔎 Variables",
    "ℹ️ À propos",
    "📤 Partager",
])

with tab_accueil:
    st.title("🚗 Simulateur de distance d'arrêt")
    st.markdown("---")

    st.subheader("Pourquoi ce simulateur ?")
    st.write(
        "La **distance d'arrêt** d’un véhicule dépend de plusieurs facteurs : vitesse, "
        "adhérence, temps de réaction… \n"
        "Ce simulateur vous aide à visualiser et comprendre leur impact. "
    )

    st.info(
        "ℹ️ En France, la distance d’arrêt réglementaire à **50 km/h** est d’environ **25 m**. "
    )

    st.markdown("### Qu'est-ce que la distance d'arrêt ?")
    st.write(
        "- **Distance de réaction** : distance parcourue pendant votre temps de réaction\n"
        "- **Distance de freinage** : distance parcourue le temps que les freins agissent\n\n"
        "La somme des deux donne la **distance d'arrêt**."
    )

# --------------------------------------------------------------
# 5. Exécution / affichage
# --------------------------------------------------------------
if "dist" not in st.session_state:
    st.session_state["dist"] = None
    st.session_state["params"] = None

if run_sim:
    t0 = time.time()
    with st.spinner("Simulation en cours..."):
        dist = run_mc(params)
    dt = time.time() - t0
    st.session_state["dist"] = dist
    st.session_state["params"] = params
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

    sorted_dist = np.sort(dist)
    x_max = max(child_d, sorted_dist.max())

    # -------- Graphiques et KPIs --------------------------------------
    with tab_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance moyenne (m)", f"{mean:.1f}")
        c2.metric("Écart type (m)", f"{std:.1f}")
        c3.metric(f"Distance P{p_val} (m)", f"{p_quant:.1f}")
        prob_color = "🟢" if p_coll < 0.01 else ("🟠" if p_coll < 0.10 else "🔴")
        c4.metric("Probabilité de collision", f"{p_coll*100:.1f} % {prob_color}")

        fig_hist = px.histogram(
            dist,
            nbins=60,
            histnorm="percent",
            labels={
                "value": "Distance d'arrêt (m)",
                "percent": "Part des simulations (%)",
            },
            template="plotly_white",
        )
        fig_hist.update_traces(name="Simulation")
        fig_hist.update_layout(
            title={"text": "Répartition des distances d'arrêt observées", "x": 0},
            title_font_size=18,
            xaxis_title="Distance d'arrêt (m)",
            yaxis_title="Part des simulations (%)",
            legend_title_text="",
            showlegend=False,
            annotations=[],
            margin=dict(b=80),
            meta={
                "description": (
                    "Histogramme montrant la distribution des distances "
                    "d'arrêt en pourcentage des simulations."
                )
            },
        )
        fig_hist.add_vline(
            x=child_d,
            line_dash="dash",
            line_width=2,
            annotation_text="Position de l'obstacle",
            annotation_position="top",
        )
        tick_step = nice_ticks(x_max)
        fig_hist.update_xaxes(range=[0, x_max], dtick=tick_step)
        st.plotly_chart(fig_hist, use_container_width=True)

        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        fig_cdf = px.area(
            x=sorted_dist,
            y=cdf * 100,
            labels={"x": "Distance d'arrêt (m)", "y": "Probabilité cumulée (%)"},
            template="plotly_white",
        )
        fig_cdf.update_traces(name="CDF")
        fig_cdf.update_yaxes(range=[0, 100])
        fig_cdf.update_xaxes(range=[0, x_max], dtick=tick_step)
        fig_cdf.add_vline(
            x=child_d,
            line_dash="dash",
            line_width=2,
            annotation_text="Position de l'obstacle",
            annotation_position="top",
        )
        fig_cdf.add_vrect(
            x0=child_d,
            x1=x_max,
            fillcolor="red",
            opacity=0.2,
            line_width=0,
            layer="below",
        )
        fig_cdf.add_annotation(
            x=child_d,
            y=0.95,
            xref="x",
            yref="paper",
            xanchor="left",
            text=f"{p_coll*100:.1f} %",
            showarrow=False,
            font=dict(color="red"),
        )
        fig_cdf.update_layout(
            title={"text": "Probabilité qu’un véhicule atteigne l'obstacle avant l’arrêt", "x": 0},
            xaxis_title="Distance d'arrêt (m)",
            yaxis_title="Probabilité cumulée (%)",
            legend_title_text="",
            showlegend=True,
            annotations=[],
            margin=dict(b=80),
            meta={
                "description": (
                    "Courbe de probabilité cumulée de collision selon la distance d'arrêt."
                )
            },
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

        st.caption(
            f"{format(len(dist), ',').replace(',', '\u202f')} tirages – {dt:.2f}s"
        )

    # -------- Statistiques --------------------------------------------
    with tab_stats:
        st.subheader("Statistiques")
        st.write(
            f"La distance d'arrêt moyenne est **{mean:.1f} ± {ci:.1f} m** "
            f"(niveau de confiance {params.conf*100:.0f} %)."
        )
        chance = math.inf if p_coll == 0 else int(round(1 / p_coll))
        if chance != math.inf:
            st.write(
                f"Avec ces conditions, il y a **1 chance sur {chance}** de ne pas s'arrêter avant l'obstacle."
            )
        else:
            st.write("La collision est quasi impossible.")
        q25, q50, q75 = np.percentile(dist, [25, 50, 75])
        max_theo = stopping_distance(
            np.array([params.speed]),
            np.array([3.0]),
            np.array([mu_bounds(base_mu(params.surface, params.tyre))[0]]),
            np.array([SLOPE[params.slope] - 1]),
        )[0]
        st.markdown(
            f"Minimum : {dist.min():.1f} m  \n"
            f"1er quartile : {q25:.1f} m  \n"
            f"Médiane : {q50:.1f} m  \n"
            f"3e quartile : {q75:.1f} m  \n"
            f"Maximum : {dist.max():.1f} m  \n"
            f"Maximum théorique : {max_theo:.1f} m"
        )
        fig_box = px.box(
            dist,
            points=False,
            labels={"value": "Distance d'arrêt (m)"},
            template="plotly_white",
            title="Boîte à moustaches"
        )
        fig_box.update_layout(
            xaxis_title="Distance d'arrêt (m)",
            yaxis_title="Valeur",
            showlegend=True,
        )
        fig_box.update_traces(name="Simulation")
        st.plotly_chart(fig_box, use_container_width=True)

    # -------- Distributions internes ----------------------------------
    with tab_var:
        st.subheader("Distributions internes")
        rng = RNG
        
        xs = np.linspace(speed_params(speed)[0], speed, 300)
        a, c, b = speed_params(speed)
        data = sample_speed(speed, 10_000, rng)
        fig = px.histogram(
            data,
            nbins=40,
            histnorm="probability density",
            opacity=0.6,
            template="plotly_white",
        )
        fig.update_traces(name="Simulation")
        fig.add_scatter(x=xs, y=speed_pdf(xs, speed), name="Densité théorique")
        fig.add_annotation(
            text=f"Loi triangulaire : a={a:.1f} c={c:.1f} b={b:.1f}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
            xanchor="center",
            align="center",
        )
        fig.update_layout(
            title_text="Vitesse réelle (km/h)",
            title_x=0.5,
            margin=dict(t=80),
            xaxis_title="Vitesse réelle (km/h)",
            yaxis_title="Densité (%)",
            legend=dict(
                title="Courbes",
                orientation="v",
                x=1.02,
                y=1,
                bordercolor="black",
                borderwidth=1,
            ),
            meta={
                "description": (
                    "Histogramme simulé et densité théorique de la vitesse réelle."
                )
            },
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        xs = np.linspace(0.3, 3, 300)
        lam = weib_scale(PROFILE_MED[profile])
        data = sample_tr(profile, 10_000, rng)
        fig = px.histogram(
            data,
            nbins=40,
            histnorm="probability density",
            opacity=0.6,
            template="plotly_white",
        )
        fig.update_traces(name="Simulation")
        fig.add_scatter(x=xs, y=tr_pdf(xs, profile), name="Densité théorique")
        fig.add_annotation(
            text=f"Weibull tronquée : λ={lam:.2f} k={K_WEIB}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
            xanchor="center",
            align="center",
        )
        fig.update_layout(
            title_text="Temps de réaction (s)",
            title_x=0.5,
            margin=dict(t=80),
            xaxis_title="Temps de réaction (s)",
            yaxis_title="Densité (%)",
            legend=dict(
                title="Courbes",
                orientation="v",
                x=1.02,
                y=1,
                bordercolor="black",
                borderwidth=1,
            ),
            meta={
                "description": (
                    "Histogramme simulé et densité théorique du temps de réaction."
                )
            },
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

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
        fig.update_traces(name="Simulation")
        fig.add_scatter(x=xs, y=mu_pdf(xs, surface, tyre), name="Densité théorique")
        fig.add_annotation(
            text=(
                f"Bêta bornée : α={A_B} β={B_B}"
                f"\n[{μ_min:.2f}, {μ_max:.2f}]"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
            xanchor="center",
            align="center",
        )
        fig.update_layout(
            title_text="Coefficient d'adhérence μ",
            title_x=0.5,
            margin=dict(t=80),
            xaxis_title="Coefficient d'adhérence μ",
            yaxis_title="Densité (%)",
            legend=dict(
                title="Courbes",
                orientation="v",
                x=1.02,
                y=1,
                bordercolor="black",
                borderwidth=1,
            ),
            meta={
                "description": (
                    "Histogramme simulé et densité théorique du coefficient d'adhérence."
                )
            },
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

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
        fig.update_traces(name="Simulation")
        fig.add_scatter(x=xs, y=theta_pdf(xs, slope), name="Densité théorique")
        fig.add_annotation(
            text=(
                f"Normale tronquée : μ={μθ:+.1f} σ=0.5"
                f"\n[{μθ-1:.1f}, {μθ+1:.1f}]"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,
            showarrow=False,
            xanchor="center",
            align="center",
        )
        fig.update_layout(
            title_text="Angle d'inclinaison de la route θ (°)",
            title_x=0.5,
            margin=dict(t=80),
            xaxis_title="Angle d'inclinaison de la route θ (°)",
            yaxis_title="Densité (%)",
            legend=dict(
                title="Courbes",
                orientation="v",
                x=1.02,
                y=1,
                bordercolor="black",
                borderwidth=1,
            ),
            meta={
                "description": (
                    "Histogramme simulé et densité théorique de l'inclinaison de la route."
                )
            },
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

else:
    tab_res.info("Aucun résultat pour l'instant.")
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
                • **État de la route :** {saved.surface}<br>
                • **État des pneus :** {saved.tyre}<br>
                • **Adhérence nominale μ :** {mu_base:.2f} (plage simulée ±0,15)<br>
                • **Inclinaison de la route :** {SLOPE[saved.slope]:+} ° ({saved.slope})<br>
                • **Confiance MC :** {saved.conf*100:.0f} %<br>
                • **Distance obstacle :** {saved.child_d} m
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info("Aucune simulation pour l'instant.")

    st.markdown("### Auteurs")
    st.markdown(
        "Ce calculateur a été réalisé par **Mohamed MOSTEFAOUI** "
        "et **Elouan LE GUYADER** – Toulouse INP‑ENSIACET."
    )

# ------------------ Partager ----------------------------------
with tab_share:
    st.markdown("### Partager l'application")
    share_url = "https://distance-arret.streamlit.app/"
    qr_img = qrcode.make(share_url)
    st.image(qr_img.get_image(), caption=share_url, use_container_width=False)
    st.write("Scannez ce QR code pour accéder au simulateur.")
