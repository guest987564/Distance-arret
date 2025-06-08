"""Streamlit front-end for the stopping distance simulator."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
import plotly.express as px

from model import SimParams, run_mc
from sampling import (
    PROFILE_MED,
    SURFACE_μ,
    SLOPE,
    speed_params,
    speed_pdf,
    sample_speed,
    tr_pdf,
    sample_tr,
    mu_pdf,
    sample_mu,
    mu_bounds,
    base_mu,
    theta_pdf,
    sample_theta,
)

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
        rng = np.random.default_rng(42)
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
