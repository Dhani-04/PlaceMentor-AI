"""PlaceMentor AI: student skill profiling with K-Means clustering (Streamlit app)."""
from __future__ import annotations

import html as _html
import sys
import textwrap
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.clustering import (
    CAREER_READY,
    DEVELOPING,
    PRIORITY,
    add_clusters,
    cluster_summary,
    ensure_project_assets,
    evaluate_k_range,
    fit_pca,
    predict_profile,
)
from utils.preprocessing import (
    FEATURE_LABELS,
    FEATURE_RANGES,
    FEATURES,
    validate_user_input,
)
from utils.recommendations import (
    AREAS,
    BENCHMARKS,
    GROUP_SUMMARIES,
    analyse_skill_gaps,
)

st.set_page_config(page_title="PlaceMentor AI", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
ICE = "#EEF0F4"
AUBERGINE = "#382442"
MARIGOLD = "#F2B83F"
WHITE = "#FFFFFF"
SOFT_GREY = "#DCE0E7"
MUTED = "#746B78"
AUBERGINE_TINT = "#9A82A6"  # lighter tint of the aubergine, used for the third group

PROFILE_COLORS = {CAREER_READY: AUBERGINE, DEVELOPING: MARIGOLD, PRIORITY: AUBERGINE_TINT}
PROFILE_ORDER = [CAREER_READY, DEVELOPING, PRIORITY]

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th {{
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
}}
.stApp {{ background: {ICE}; color: {AUBERGINE}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}
h1, h2, h3, h4 {{ color: {AUBERGINE}; letter-spacing: -0.01em; }}
h3 {{ font-weight: 700; margin-top: 1.6rem; }}
.stApp p, .stApp li {{ color: {AUBERGINE}; }}
[data-testid="stCaptionContainer"], .pm-muted {{ color: {MUTED} !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {AUBERGINE}; border-right: none; }}
[data-testid="stSidebar"] * {{ color: {ICE}; }}
[data-testid="stSidebarNav"]::before {{
    content: "PlaceMentor AI"; display: block; padding: 1.4rem 1.2rem .6rem;
    font-weight: 800; font-size: 1.35rem; color: {MARIGOLD};
}}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{
    border-radius: 10px; margin: 2px 10px;
}}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover {{
    background: rgba(238, 240, 244, 0.12);
}}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: {MARIGOLD}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNavLink"][aria-current="page"] * {{ color: {AUBERGINE} !important; font-weight: 700; }}
.pm-sidepanel {{
    margin: 1rem 12px 0; padding: 1rem; border-radius: 14px;
    background: rgba(238, 240, 244, 0.08); border: 1px solid rgba(238, 240, 244, 0.16);
    font-size: .82rem; line-height: 1.55;
}}
.pm-sidepanel b {{ color: {MARIGOLD}; }}

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
    background: {MARIGOLD}; color: {AUBERGINE}; border: 1px solid {MARIGOLD};
    border-radius: 10px; font-weight: 700; padding: .55rem 1.3rem;
    box-shadow: 0 1px 2px rgba(56, 36, 66, .12);
}}
.stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] > button p {{
    color: inherit !important; font-weight: 700;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    background: {AUBERGINE}; border-color: {AUBERGINE}; color: {MARIGOLD};
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible {{ outline: 3px solid {AUBERGINE}; outline-offset: 2px; }}

/* Inputs, forms, tables, charts */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{
    border-radius: 10px !important; background: {WHITE} !important;
}}
[data-testid="stForm"] {{
    background: {WHITE}; border: 1px solid {SOFT_GREY}; border-radius: 16px; padding: 1.4rem;
    box-shadow: 0 1px 2px rgba(56, 36, 66, .06), 0 10px 28px rgba(56, 36, 66, .05);
}}
[data-testid="stImage"] img {{
    background: {WHITE}; border: 1px solid {SOFT_GREY}; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(56, 36, 66, .06), 0 10px 28px rgba(56, 36, 66, .05);
}}
[data-testid="stDataFrame"] {{
    width: 100% !important; border: 1px solid {SOFT_GREY}; border-radius: 12px; overflow: hidden;
}}
.pm-card {{
    background: {WHITE}; border: 1px solid {SOFT_GREY}; border-radius: 16px; padding: 1.2rem 1.35rem;
    box-shadow: 0 1px 2px rgba(56, 36, 66, .06), 0 10px 28px rgba(56, 36, 66, .05); height: 100%;
}}
.pm-card h4 {{ margin: 0 0 .4rem; font-size: 1.02rem; }}
.pm-card p {{ margin: .25rem 0; font-size: .92rem; line-height: 1.6; }}
.pm-card ul, .pm-card ol {{ margin: .3rem 0 .2rem 1.1rem; padding: 0; }}
.pm-card li {{ font-size: .92rem; line-height: 1.6; margin-bottom: .2rem; }}
.pm-gap {{ height: .9rem; }}

/* Page header, hero, stats */
.pm-pagehead {{ margin-bottom: 1.2rem; }}
.pm-pagehead h1 {{ margin: 0; font-size: 1.95rem; font-weight: 800; }}
.pm-pagehead p {{ margin: .35rem 0 0; color: {MUTED}; font-size: 1rem; max-width: 62ch; }}
.pm-pagehead::after {{ content: ""; display: block; width: 52px; height: 4px; border-radius: 4px; background: {MARIGOLD}; margin-top: .8rem; }}
.pm-hero {{
    background: {AUBERGINE}; border-radius: 22px; padding: 2.2rem 2.4rem; margin-bottom: 1.3rem;
    box-shadow: 0 14px 36px rgba(56, 36, 66, .22);
}}
.pm-hero h1, .pm-hero p, .pm-hero span {{ color: {ICE}; }}
.pm-hero h1 {{ margin: 0; font-size: 2.5rem; font-weight: 800; color: {WHITE}; }}
.pm-hero .pm-hero-title {{ color: {MARIGOLD}; font-weight: 700; font-size: 1.1rem; margin: .4rem 0 .9rem; }}
.pm-hero .pm-hero-body {{ max-width: 64ch; line-height: 1.7; margin-bottom: 1.1rem; }}
.pm-badge {{
    display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 700;
    background: {MARIGOLD}; color: {AUBERGINE} !important; margin: 0 .5rem .4rem 0;
}}
.pm-badge.ghost {{ background: transparent; border: 1px solid rgba(238, 240, 244, .45); color: {ICE} !important; }}
.pm-stat {{ padding: 1.1rem 1.25rem; border-left: 5px solid {AUBERGINE}; }}
.pm-stat.key {{ background: {MARIGOLD}; border-color: {AUBERGINE}; }}
.pm-stat-label {{ font-size: .84rem; font-weight: 600; color: {MUTED}; }}
.pm-stat.key .pm-stat-label {{ color: {AUBERGINE}; }}
.pm-stat-value {{ font-size: 2rem; font-weight: 800; line-height: 1.15; margin: .15rem 0; color: {AUBERGINE}; }}
.pm-stat-note {{ font-size: .78rem; color: {MUTED}; }}
.pm-stat.key .pm-stat-note {{ color: {AUBERGINE}; }}

/* Tables */
.pm-table {{
    background: {WHITE}; border: 1px solid {SOFT_GREY}; border-radius: 14px; overflow-x: auto;
    box-shadow: 0 1px 2px rgba(56, 36, 66, .06);
}}
.pm-table table {{ width: 100%; border-collapse: collapse; }}
.pm-table th {{
    background: {AUBERGINE}; color: {WHITE}; text-align: left !important; padding: .65rem .85rem;
    font-size: .8rem; font-weight: 600; white-space: nowrap;
}}
.pm-table td {{ padding: .6rem .85rem; border-bottom: 1px solid {SOFT_GREY}; font-size: .88rem; color: {AUBERGINE}; }}
.pm-table tr:last-child td {{ border-bottom: none; }}
.pm-table tbody tr:nth-child(even) td {{ background: {ICE}; }}

/* Recommendation cards and chips */
.pm-rec {{
    background: {WHITE}; border: 1px solid {SOFT_GREY}; border-left: 6px solid {SOFT_GREY};
    border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .7rem;
}}
.pm-rec.priority {{ border-left-color: {AUBERGINE}; }}
.pm-rec.close {{ border-left-color: {MARIGOLD}; }}
.pm-rec-head {{ display: flex; align-items: center; gap: .6rem; margin-bottom: .3rem; }}
.pm-rec-head b {{ font-size: .98rem; }}
.pm-rec p {{ margin: 0; font-size: .9rem; line-height: 1.6; }}
.pm-chip {{ display: inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .74rem; font-weight: 700; }}
.pm-chip.priority {{ background: {AUBERGINE}; color: {WHITE} !important; }}
.pm-chip.close {{ background: {MARIGOLD}; color: {AUBERGINE} !important; }}
.pm-chip.strong {{ background: {ICE}; color: {AUBERGINE} !important; border: 1px solid {SOFT_GREY}; }}
.pm-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: .45rem; }}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def esc(text) -> str:
    return _html.escape(str(text))


def compact(markup: str) -> str:
    """Collapse indentation so Markdown never mistakes HTML for a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def html_block(markup: str) -> None:
    st.markdown(compact(markup), unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    html_block(f'<div class="pm-pagehead"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def stat_card(label: str, value, note: str = "", key: bool = False) -> str:
    cls = "pm-card pm-stat key" if key else "pm-card pm-stat"
    return compact(f"""
        <div class="{cls}">
            <div class="pm-stat-label">{esc(label)}</div>
            <div class="pm-stat-value">{esc(value)}</div>
            <div class="pm-stat-note">{esc(note)}</div>
        </div>""")


def table_html(df: pd.DataFrame, float_format: str = "{:.2f}") -> None:
    markup = df.to_html(index=False, border=0, float_format=lambda x: float_format.format(x))
    html_block(f'<div class="pm-table">{markup}</div>')


def gap() -> None:
    html_block('<div class="pm-gap"></div>')


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def color_for(profile: str) -> str:
    return PROFILE_COLORS.get(profile, MUTED)


# --------------------------------------------------------------------------
# Load data and model (trained automatically if missing)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing dataset and K-Means model...")
def load_assets():
    return ensure_project_assets()


@st.cache_data(show_spinner="Running elbow and silhouette analysis...")
def get_k_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    return evaluate_k_range(df, 2, 8)


try:
    DF, BUNDLE, ASSET_NOTES = load_assets()
except Exception as exc:  # pragma: no cover - defensive
    st.error(
        "PlaceMentor AI could not prepare its data or model. "
        "Run `python train_model.py` in the project folder and reload this page."
    )
    st.exception(exc)
    st.stop()

CLUSTERED = add_clusters(DF, BUNDLE)
SUMMARY = cluster_summary(CLUSTERED, BUNDLE)
K = BUNDLE["k"]

st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("confirm_clear", False)
st.session_state.setdefault("sg_values", None)


# --------------------------------------------------------------------------
# Chart helpers (matplotlib, themed)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": AUBERGINE,
    "axes.labelcolor": AUBERGINE,
    "axes.titlecolor": AUBERGINE,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
})


def style_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SOFT_GREY)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis=grid_axis, color=SOFT_GREY, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def show_fig(fig) -> None:
    fig.tight_layout(pad=1.3)
    st.pyplot(fig)
    plt.close(fig)


def profile_legend(ax_or_fig, profiles, loc="best") -> None:
    handles = [Patch(facecolor=color_for(p), edgecolor="none", label=p) for p in profiles]
    ax_or_fig.legend(handles=handles, frameon=False, fontsize=8.5, loc=loc, labelcolor=AUBERGINE)


def chart_cluster_sizes() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=130)
    labels = [f"{textwrap.fill(r['Profile Group'], 14)}\n(Cluster {cid})" for cid, r in SUMMARY.iterrows()]
    colors = [color_for(r["Profile Group"]) for _, r in SUMMARY.iterrows()]
    bars = ax.bar(labels, SUMMARY["Students"], color=colors, width=0.55)
    for bar, (_, r) in zip(bars, SUMMARY.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{int(r['Students'])}  ({r['Share (%)']:.0f}%)", ha="center", fontsize=9, color=AUBERGINE, fontweight="bold")
    ax.set_ylabel("Students")
    ax.set_ylim(0, SUMMARY["Students"].max() * 1.18)
    style_axes(ax)
    show_fig(fig)


def chart_pca(user_point=None, size=(6.2, 4.4)) -> None:
    pca, coords = fit_pca(DF, BUNDLE)
    fig, ax = plt.subplots(figsize=size, dpi=130)
    for profile in PROFILE_ORDER:
        mask = (CLUSTERED["Profile Group"] == profile).to_numpy()
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=34, color=color_for(profile),
                       alpha=0.85, edgecolor=WHITE, linewidth=0.5, label=profile)
    cent = pca.transform(BUNDLE["centroids_scaled"])
    for cid in range(K):
        ax.scatter(cent[cid, 0], cent[cid, 1], marker="X", s=170, color=color_for(BUNDLE["cluster_labels"][cid]),
                   edgecolor=WHITE, linewidth=1.6, zorder=5)
    ax.scatter([], [], marker="X", s=90, color=MUTED, edgecolor=WHITE, label="Centroid")
    if user_point is not None:
        up = pca.transform(user_point.reshape(1, -1))[0]
        ax.scatter(up[0], up[1], marker="D", s=120, color=WHITE, edgecolor=AUBERGINE, linewidth=2.2, zorder=6, label="Your profile")
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Principal component 1 ({var[0]:.1f}% of variance)")
    ax.set_ylabel(f"Principal component 2 ({var[1]:.1f}% of variance)")
    style_axes(ax, "both")
    ax.legend(frameon=False, fontsize=8.5, loc="best", labelcolor=AUBERGINE)
    show_fig(fig)


def chart_feature_grid() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), dpi=130)
    colors = [color_for(r["Profile Group"]) for _, r in SUMMARY.iterrows()]
    xlabels = [f"C{cid}" for cid in SUMMARY.index]
    for ax, feat in zip(axes.flat, FEATURES):
        vals = SUMMARY[feat].to_numpy()
        bars = ax.bar(xlabels, vals, color=colors, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}" if feat != "CGPA" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=AUBERGINE)
        ax.set_title(feat, fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylim(0, vals.max() * 1.22)
        style_axes(ax)
    for ax in list(axes.flat)[len(FEATURES):]:
        ax.axis("off")
    present = [r["Profile Group"] for _, r in SUMMARY.iterrows()]
    axes.flat[-1].legend(handles=[Patch(facecolor=color_for(p), label=f"{p}") for p in present],
                         frameon=False, fontsize=9, loc="center", labelcolor=AUBERGINE)
    show_fig(fig)


def chart_elbow(ev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=130)
    ax.plot(ev["k"], ev["inertia"], color=AUBERGINE, linewidth=2.2, marker="o", markersize=6, markerfacecolor=WHITE,
            markeredgewidth=2)
    row = ev[ev["k"] == K]
    if not row.empty:
        ax.scatter(row["k"], row["inertia"], s=130, color=MARIGOLD, edgecolor=AUBERGINE, linewidth=1.6, zorder=5,
                   label=f"Chosen k = {K}")
        ax.legend(frameon=False, fontsize=9, labelcolor=AUBERGINE)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_xticks(ev["k"])
    style_axes(ax)
    show_fig(fig)


def chart_silhouette(ev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=130)
    colors = [MARIGOLD if k == K else AUBERGINE for k in ev["k"]]
    bars = ax.bar(ev["k"].astype(str), ev["silhouette"], color=colors, width=0.6)
    for bar, v in zip(bars, ev["silhouette"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5, color=AUBERGINE)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_ylim(0, max(ev["silhouette"].max() * 1.2, 0.1))
    style_axes(ax)
    show_fig(fig)


def chart_student_vs_group(values: dict, group_avg: pd.Series, profile: str) -> None:
    """Both bars are shown as a percentage of each feature's maximum possible value."""
    pct_student = [values[f] / FEATURE_RANGES[f][1] * 100 for f in FEATURES]
    pct_group = [group_avg[f] / FEATURE_RANGES[f][1] * 100 for f in FEATURES]
    x = np.arange(len(FEATURES))
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=130)
    ax.bar(x - 0.19, pct_student, width=0.38, color=MARIGOLD, label="You")
    ax.bar(x + 0.19, pct_group, width=0.38, color=AUBERGINE, label=f"{profile} (average)")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, fontsize=8.5)
    ax.set_ylabel("Position on scale (% of maximum)")
    ax.set_ylim(0, 112)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", labelcolor=AUBERGINE)
    show_fig(fig)


def chart_gap(items: list[dict]) -> None:
    ordered = list(reversed(items))
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=130)
    tier_color = {"priority": AUBERGINE, "close": MARIGOLD, "strong": AUBERGINE_TINT}
    vals = [r["Percent of benchmark"] for r in ordered]
    bars = ax.barh([r["Area"] for r in ordered], vals, color=[tier_color[r["Tier"]] for r in ordered], height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 2, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8.5, color=AUBERGINE)
    ax.axvline(100, color=MUTED, linestyle="--", linewidth=1.3)
    ax.set_xlim(0, max(max(vals) * 1.15, 125))
    ax.set_xlabel("Your value as a percentage of the illustrative benchmark")
    style_axes(ax, "x")
    handles = [Patch(facecolor=AUBERGINE, label="Needs focus"), Patch(facecolor=MARIGOLD, label="Close to benchmark"),
               Patch(facecolor=AUBERGINE_TINT, label="Meets benchmark"),
               Line2D([0], [0], color=MUTED, linestyle="--", linewidth=1.3, label="Benchmark (100%)")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=4, labelcolor=AUBERGINE)
    show_fig(fig)


# --------------------------------------------------------------------------
# Shared input form (used by Student Assessment and Skill Gap Analysis)
# --------------------------------------------------------------------------
DEFAULTS = {"CGPA": 7.5, "Aptitude": 65.0, "Coding": 60.0, "Communication": 60.0,
            "Projects": 2, "Internships": 1, "Attendance": 80.0}


def input_form(prefix: str, button_label: str, ask_name: bool = True):
    """Render the 8-field form. Returns (submitted, name, values)."""
    with st.form(f"{prefix}_form"):
        name = ""
        if ask_name:
            name = st.text_input("Student name (optional)", max_chars=50, key=f"{prefix}_name",
                                 placeholder="Leave blank to stay anonymous")
        c1, c2, c3 = st.columns(3)
        values = {}
        with c1:
            values["CGPA"] = st.number_input(FEATURE_LABELS["CGPA"], 0.0, 10.0, DEFAULTS["CGPA"], 0.01,
                                             format="%.2f", key=f"{prefix}_cgpa")
            values["Communication"] = st.number_input(FEATURE_LABELS["Communication"], 0.0, 100.0,
                                                      DEFAULTS["Communication"], 1.0, format="%.1f",
                                                      key=f"{prefix}_comm")
            values["Internships"] = st.number_input(FEATURE_LABELS["Internships"], 0, 5, DEFAULTS["Internships"], 1,
                                                    key=f"{prefix}_intern")
        with c2:
            values["Aptitude"] = st.number_input(FEATURE_LABELS["Aptitude"], 0.0, 100.0, DEFAULTS["Aptitude"], 1.0,
                                                 format="%.1f", key=f"{prefix}_apt")
            values["Projects"] = st.number_input(FEATURE_LABELS["Projects"], 0, 10, DEFAULTS["Projects"], 1,
                                                 key=f"{prefix}_proj")
            values["Attendance"] = st.number_input(FEATURE_LABELS["Attendance"], 0.0, 100.0, DEFAULTS["Attendance"],
                                                   1.0, format="%.1f", key=f"{prefix}_att")
        with c3:
            values["Coding"] = st.number_input(FEATURE_LABELS["Coding"], 0.0, 100.0, DEFAULTS["Coding"], 1.0,
                                               format="%.1f", key=f"{prefix}_code")
        submitted = st.form_submit_button(button_label)
    return submitted, name.strip(), values


def recommendation_cards(analysis: dict, limit: int | None = None) -> None:
    rows = analysis["prioritised"][:limit] if limit else analysis["prioritised"]
    for r in rows:
        html_block(f"""
            <div class="pm-rec {r['Tier']}">
                <div class="pm-rec-head"><b>{esc(r['Area'])}</b>
                <span class="pm-chip {r['Tier']}">{esc(r['Status'])}</span></div>
                <p>{esc(r['Suggestion'])}</p>
            </div>""")


# --------------------------------------------------------------------------
# PAGE 1: Overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    html_block(f"""
        <div class="pm-hero">
            <h1>PlaceMentor AI</h1>
            <p class="pm-hero-title">Student skill profiling and placement readiness analysis using K-Means clustering</p>
            <p class="pm-hero-body">Welcome. Compare an academic and skill profile with {len(DF)} simulated student
            profiles, see which group it resembles, and get practical steps to strengthen the areas that matter most.</p>
            <span class="pm-badge">Simulated dataset</span>
            <span class="pm-badge ghost">Exploratory clustering, not a placement prediction</span>
        </div>""")

    for note in ASSET_NOTES:
        st.info(note)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Student profiles", len(DF), "simulated records"), unsafe_allow_html=True)
    c2.markdown(stat_card("Clusters", K, "K-Means groups"), unsafe_allow_html=True)
    c3.markdown(stat_card("Silhouette score", f"{BUNDLE['silhouette']:.3f}", "current model", key=True),
                unsafe_allow_html=True)
    c4.markdown(stat_card("Student features", len(FEATURES), "academic and skill indicators"),
                unsafe_allow_html=True)

    st.markdown("### Student cluster distribution")
    left, right = st.columns([1, 1.15])
    with left:
        chart_cluster_sizes()
    with right:
        chart_pca(size=(6.2, 3.9))

    st.markdown("### Explore the modules")
    items = [
        ("student_assessment", "Student Assessment", "Enter a profile and see the nearest group."),
        ("cluster_explorer", "Cluster Explorer", "Compare group averages and browse student records."),
        ("model_evaluation", "Model Evaluation", "Elbow, silhouette, PCA and centroids."),
        ("skill_gap", "Skill Gap Analysis", "See where a profile sits against development benchmarks."),
        ("student_history", "Student History", "Review and export this session's assessments."),
        ("about_project", "About the Project", "Method, dataset, limitations and next steps."),
    ]
    cols = st.columns(3)
    for i, (key, title, desc) in enumerate(items):
        with cols[i % 3]:
            html_block(f'<div class="pm-card"><h4>{esc(title)}</h4><p class="pm-muted">{esc(desc)}</p></div>')
            if st.button(f"Open {title}", key=f"qn_{key}"):
                st.switch_page(PAGES[key])
            gap()


# --------------------------------------------------------------------------
# PAGE 2: Student Assessment
# --------------------------------------------------------------------------
def page_assessment() -> None:
    page_header("Student Assessment",
                "Enter academic and skill indicators to find the group of simulated student profiles this one resembles.")

    submitted, name, values = input_form("as", "ANALYSE MY PROFILE")

    if submitted:
        errors = validate_user_input(values)
        if errors:
            for err in errors:
                st.error(err)
        else:
            result = predict_profile(BUNDLE, values)
            analysis = analyse_skill_gaps(values)
            record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Name": name or "Anonymous"}
            record.update({f: values[f] for f in FEATURES})
            record.update({"Cluster": result["cluster"], "Profile Group": result["profile"],
                           "Centroid Distance": round(result["distance"], 3)})
            st.session_state["history"].append(record)
            st.session_state["last_result"] = {"name": name or "Anonymous", "values": values,
                                               "result": result, "analysis": analysis}
            st.success("Profile analysed and saved to this session's history.")

    last = st.session_state.get("last_result")
    if not last:
        st.info("Fill in the form and press ANALYSE MY PROFILE to see your cluster, group and recommendations.")
        return

    result, values, analysis = last["result"], last["values"], last["analysis"]
    profile = result["profile"]

    st.markdown(f"### Result for {esc(last['name'])}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("Assigned cluster", f"Cluster {result['cluster']}", "nearest centroid"),
                unsafe_allow_html=True)
    c2.markdown(stat_card("Profile group", profile, "descriptive label", key=True), unsafe_allow_html=True)
    c3.markdown(stat_card("Distance from centroid", f"{result['distance']:.2f}",
                          f"typical in this group: {result['typical_distance']:.2f}"), unsafe_allow_html=True)
    gap()
    html_block(f"""
        <div class="pm-card">
            <h4><span class="pm-dot" style="background:{color_for(profile)}"></span>{esc(profile)}</h4>
            <p>{esc(GROUP_SUMMARIES[profile])}</p>
            <p class="pm-muted">{esc(result['closeness'])}. Distances are measured in standardised feature units.
            Cluster membership describes similarity to simulated student profiles. It does not predict placement
            selection or success.</p>
        </div>""")

    st.markdown("### Your profile compared with the group")
    left, right = st.columns([1.1, 1])
    group_avg = CLUSTERED[CLUSTERED["Cluster"] == result["cluster"]][FEATURES].mean()
    with left:
        chart_student_vs_group(values, group_avg, profile)
    with right:
        chart_pca(user_point=result["scaled"], size=(5.4, 3.8))

    st.markdown("### Skill-development recommendations")
    if analysis["focus"]:
        st.caption("Suggested focus areas, in order: " + ", ".join(analysis["focus"]) + ".")
    else:
        st.caption("Every value meets the illustrative benchmarks. Keep building depth.")
    recommendation_cards(analysis, limit=4)
    st.caption("Open Skill Gap Analysis in the sidebar for the full breakdown of all seven areas.")


# --------------------------------------------------------------------------
# PAGE 3: Cluster Explorer
# --------------------------------------------------------------------------
def page_clusters() -> None:
    page_header("Cluster Explorer", "See how the K-Means groups differ and browse the students in each one.")

    cols = st.columns(K)
    for col, (cid, row) in zip(cols, SUMMARY.iterrows()):
        col.markdown(stat_card(f"Cluster {cid}: {row['Profile Group']}", int(row["Students"]),
                               f"{row['Share (%)']:.1f}% of students", key=row["Profile Group"] == CAREER_READY),
                     unsafe_allow_html=True)

    st.markdown("### Cluster-wise averages")
    table = SUMMARY.reset_index().rename(columns={"Cluster": "Cluster ID"}).drop(columns=["Composite"])
    table_html(table)
    st.caption("Profile names are descriptive labels assigned after clustering by comparing the centroids. "
               "K-Means itself does not learn these labels.")

    st.markdown("### Group size and comparison charts")
    left, right = st.columns([1, 1.6])
    with left:
        chart_cluster_sizes()
    with right:
        st.caption("Average of each feature in every cluster (C0, C1, ... are cluster IDs).")
        chart_feature_grid()

    st.markdown("### How the names were assigned")
    comp = SUMMARY.reset_index()[["Cluster", "Profile Group", "Composite"]].rename(
        columns={"Cluster": "Cluster ID", "Composite": "Mean standardised centroid score"})
    table_html(comp)
    st.caption("The cluster with the highest mean standardised centroid value is called Career-ready profile, the "
               "lowest is Skill-building priority and the remaining one is Developing talent.")

    st.markdown("### Student records by cluster")
    options = [int(c) for c in SUMMARY.index]
    chosen = st.selectbox("Choose a cluster", options, key="explorer_cluster",
                          format_func=lambda c: f"Cluster {c}: {BUNDLE['cluster_labels'][c]}")
    records = CLUSTERED[CLUSTERED["Cluster"] == chosen].copy()
    records.insert(0, "Student #", records.index + 1)
    st.dataframe(records.drop(columns=["Cluster"]), hide_index=True, height=360)
    st.caption(f"{len(records)} students in this cluster.")

    st.download_button("Download clustered dataset (CSV)", csv_bytes(CLUSTERED),
                       file_name="clustered_students.csv", mime="text/csv", key="dl_clustered")


# --------------------------------------------------------------------------
# PAGE 4: ML Model Evaluation
# --------------------------------------------------------------------------
def silhouette_reading(score: float) -> str:
    if score >= 0.50:
        return "well-separated clusters"
    if score >= 0.25:
        return "moderately separated clusters with some overlap between groups"
    return "weakly separated clusters with heavy overlap"


def page_evaluation() -> None:
    page_header("ML Model Evaluation",
                "Internal clustering measures for the K-Means model. No ground-truth labels exist, so accuracy-style "
                "metrics do not apply.")
    ev = get_k_evaluation(DF)
    pca, _ = fit_pca(DF, BUNDLE)
    var = pca.explained_variance_ratio_ * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Clusters used", K, "final model"), unsafe_allow_html=True)
    c2.markdown(stat_card("Silhouette score", f"{BUNDLE['silhouette']:.3f}", "final model", key=True),
                unsafe_allow_html=True)
    c3.markdown(stat_card("Inertia (WCSS)", f"{BUNDLE['inertia']:.1f}", "final model"), unsafe_allow_html=True)
    c4.markdown(stat_card("PCA variance shown", f"{var.sum():.1f}%", "first two components"),
                unsafe_allow_html=True)

    st.markdown("### Elbow method and silhouette score")
    left, right = st.columns(2)
    with left:
        st.caption("Inertia for k = 2 to 8. The curve bends where extra clusters stop reducing it much.")
        chart_elbow(ev)
    with right:
        st.caption("Silhouette score for k = 2 to 8. Higher means better-separated clusters.")
        chart_silhouette(ev)
    table_html(ev.rename(columns={"k": "k", "inertia": "Inertia (WCSS)", "silhouette": "Silhouette score"}),
               float_format="{:.3f}")

    best = ev.loc[ev["silhouette"].idxmax()]
    chosen = ev[ev["k"] == K].iloc[0]
    reading = silhouette_reading(float(chosen["silhouette"]))
    explain = (f"With k = {K} the silhouette score is {chosen['silhouette']:.3f}, which indicates {reading}. "
               f"Inertia falls from {ev['inertia'].iloc[0]:.0f} at k = {int(ev['k'].iloc[0])} to "
               f"{ev['inertia'].iloc[-1]:.0f} at k = {int(ev['k'].iloc[-1])}, and the gains shrink after the first few "
               f"clusters.")
    if int(best["k"]) != K:
        explain += (f" The highest silhouette in the tested range is at k = {int(best['k'])} "
                    f"({best['silhouette']:.3f}), but k = {K} was kept because three groups give more useful, "
                    f"interpretable profile levels (strong, developing and early-stage) than {int(best['k'])}.")
    else:
        explain += f" k = {K} also has the highest silhouette in the tested range."
    html_block(f'<div class="pm-card"><h4>Interpretation</h4><p>{esc(explain)}</p></div>')

    st.markdown("### PCA visualisation")
    left, right = st.columns([1.3, 1])
    with left:
        chart_pca(size=(6.6, 4.5))
    with right:
        html_block(f"""
            <div class="pm-card"><h4>Reading the plot</h4>
            <p>PCA compresses the seven standardised features into two axes for display only. The two components keep
            {var[0]:.1f}% and {var[1]:.1f}% of the variance. Clusters are fitted on all seven features, not on these
            two axes, so some overlap on the plot is expected. X marks each centroid.</p></div>""")

    st.markdown("### Cluster centroids and sizes")
    names = [BUNDLE["cluster_labels"][c] for c in range(K)]
    orig = pd.DataFrame(BUNDLE["centroids_original"], columns=FEATURES)
    orig.insert(0, "Profile Group", names)
    orig.insert(0, "Cluster", range(K))
    orig.insert(2, "Students", [int((CLUSTERED["Cluster"] == c).sum()) for c in range(K)])
    st.caption("Centroids in original units")
    table_html(orig)
    gap()
    scaled = pd.DataFrame(BUNDLE["centroids_scaled"], columns=FEATURES)
    scaled.insert(0, "Cluster", range(K))
    st.caption("Centroids in standardised units (0 = dataset average)")
    table_html(scaled)

    st.markdown("### Methodology notes")
    html_block("""
        <div class="pm-card"><ul>
        <li>The StandardScaler and K-Means are stored together in one pipeline, so new student input is transformed
        with the training mean and standard deviation. The scaler is never refitted on new input, which avoids
        leakage and preprocessing mismatch.</li>
        <li>Clustering is unsupervised: there are no true labels, so accuracy, precision, recall and confusion
        matrices are not reported.</li>
        <li>Inertia, silhouette score and PCA are internal measures. They describe how compact and separated the
        groups are, not whether the groups predict placement outcomes.</li>
        <li>The dataset is simulated, so these scores describe this artificial data only.</li></ul></div>""")


# --------------------------------------------------------------------------
# PAGE 5: Skill Gap Analysis
# --------------------------------------------------------------------------
def page_skill_gap() -> None:
    page_header("Skill Gap Analysis",
                "Compare a profile with illustrative development benchmarks and get suggestions based on the entered values.")
    values = None
    history = st.session_state["history"]

    use_latest = False
    if history:
        source = st.radio("Which values should be analysed?", ["Latest assessment", "Enter new values"],
                          horizontal=True, key="sg_source")
        use_latest = source == "Latest assessment"

    if use_latest:
        latest = history[-1]
        values = {f: latest[f] for f in FEATURES}
        st.caption(f"Using the assessment for {latest['Name']} recorded at {latest['Timestamp']}.")
    else:
        submitted, _name, entered = input_form("sg", "Analyse skill gaps", ask_name=False)
        if submitted:
            errors = validate_user_input(entered)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.session_state["sg_values"] = entered
        values = st.session_state.get("sg_values")

    if not values:
        st.info("Enter values and press Analyse skill gaps, or complete a Student Assessment first.")
        return

    analysis = analyse_skill_gaps(values)
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("Benchmarks met", f"{analysis['met']} of {analysis['total']}",
                          "illustrative targets", key=True), unsafe_allow_html=True)
    c2.markdown(stat_card("Areas needing focus", sum(1 for r in analysis["items"] if r["Tier"] == "priority"),
                          "more than 15% below benchmark"), unsafe_allow_html=True)
    c3.markdown(stat_card("Top focus area", analysis["focus"][0] if analysis["focus"] else "None",
                          "lowest percentage of benchmark"), unsafe_allow_html=True)

    st.markdown("### Where you stand")
    left, right = st.columns([1.35, 1])
    with left:
        chart_gap(analysis["items"])
    with right:
        bench = pd.DataFrame({
            "Area": [AREAS[f] for f in FEATURES],
            "Your value": [values[f] for f in FEATURES],
            "Benchmark": [BENCHMARKS[f] for f in FEATURES],
        })
        table_html(bench, float_format="{:.2f}")

    st.markdown("### Suggestions")
    recommendation_cards(analysis)

    html_block("""
        <div class="pm-card"><h4>About the benchmarks</h4>
        <p>The benchmark values (CGPA 7.5, aptitude 65, coding 65, communication 65, 3 projects, 1 internship and
        75% attendance) are illustrative development targets chosen for this project. They are not official
        placement eligibility criteria, and companies and institutions set their own requirements.
        A value counts as needing focus when it is more than 15% below its benchmark.</p></div>""")


# --------------------------------------------------------------------------
# PAGE 6: Student History
# --------------------------------------------------------------------------
def _ask_clear() -> None:
    st.session_state["confirm_clear"] = True


def _do_clear() -> None:
    st.session_state["history"] = []
    st.session_state["last_result"] = None
    st.session_state["confirm_clear"] = False


def _cancel_clear() -> None:
    st.session_state["confirm_clear"] = False


def page_history() -> None:
    page_header("Student History",
                "Assessments recorded in this browser session. They are not saved permanently: refreshing or closing "
                "the app clears them.")
    history = st.session_state["history"]
    if not history:
        st.info("No assessments yet. Complete a Student Assessment and it will appear here.")
        if st.button("Go to Student Assessment", key="hist_go"):
            st.switch_page(PAGES["student_assessment"])
        return

    df = pd.DataFrame(history)
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("Assessments", len(df), "this session"), unsafe_allow_html=True)
    top = df["Profile Group"].value_counts().idxmax()
    c2.markdown(stat_card("Most common group", top, "in this session", key=True), unsafe_allow_html=True)
    c3.markdown(stat_card("Latest assessment", df["Name"].iloc[-1], df["Timestamp"].iloc[-1]),
                unsafe_allow_html=True)

    st.markdown("### Previously analysed profiles")
    st.dataframe(df.iloc[::-1], hide_index=True, height=380)

    a, b = st.columns([1, 3])
    with a:
        st.download_button("Export history (CSV)", csv_bytes(df), file_name="assessment_history.csv",
                           mime="text/csv", key="dl_history")
    with b:
        st.button("Clear history", key="hist_clear", on_click=_ask_clear)

    if st.session_state["confirm_clear"]:
        st.warning(f"This will permanently remove all {len(df)} assessments from this session. Continue?")
        y, n, _ = st.columns([1, 1, 4])
        y.button("Yes, clear history", key="hist_confirm", on_click=_do_clear)
        n.button("Cancel", key="hist_cancel", on_click=_cancel_clear)


# --------------------------------------------------------------------------
# PAGE 7: About
# --------------------------------------------------------------------------
def page_about() -> None:
    page_header("About the Project", "Student Skill Profiling and Placement Readiness Analysis Using K-Means Clustering.")

    def card(title: str, body: str) -> None:
        html_block(f'<div class="pm-card"><h4>{esc(title)}</h4>{body}</div>')

    card("Problem statement",
         "<p>Students often lack a structured way to understand their academic performance, technical skills and "
         "employability-related strengths. This project uses unsupervised machine learning to discover groups of "
         "students with similar academic and skill profiles and gives understandable skill-development "
         "recommendations.</p>")
    gap()
    card("Objectives",
         "<ul><li>Build an interactive application that analyses academic and professional skill indicators.</li>"
         "<li>Group similar student profiles with K-Means clustering.</li>"
         "<li>Evaluate the clustering with the elbow method, silhouette score and PCA.</li>"
         "<li>Identify improvement areas and suggest practical next steps.</li></ul>"
         "<p class='pm-muted'>The system performs exploratory profile clustering. It does not predict placement "
         "selection or probability.</p>")
    gap()
    l, r = st.columns(2)
    with l:
        card("Algorithm: K-Means",
             "<p>K-Means partitions students into k groups. It places k centroids, assigns every student to the "
             "nearest centroid, moves each centroid to the mean of its students, and repeats until assignments stop "
             "changing. It minimises inertia, the sum of squared distances from each student to their centroid.</p>"
             "<p>Features are standardised first so that CGPA (0 to 10) and scores (0 to 100) contribute fairly. "
             "A new student is assigned to the nearest centroid after the same scaling.</p>")
    with r:
        card("System workflow",
             "<ol><li>Load the student dataset and validate it.</li><li>Fill missing values and select the seven "
             "features.</li><li>Standardise with StandardScaler.</li><li>Run elbow and silhouette analysis for "
             "k = 2 to 8.</li><li>Train the final K-Means model with k = 3.</li><li>Name clusters from their "
             "averages.</li><li>Project to two dimensions with PCA.</li><li>Save the pipeline with Joblib and serve "
             "it in Streamlit.</li></ol>")
    gap()
    l, r = st.columns(2)
    with l:
        card("Technology stack",
             "<ul><li>Python</li><li>Streamlit for the interface</li><li>Pandas and NumPy for data handling</li>"
             "<li>Scikit-learn for StandardScaler, K-Means, PCA and silhouette score</li>"
             "<li>Matplotlib for charts</li><li>Joblib for model persistence</li><li>Git and GitHub</li></ul>")
    with r:
        mins = DF.min()
        maxs = DF.max()
        card("Dataset details",
             f"<p>The dataset is <b>simulated</b> and is not real institutional placement data. It holds {len(DF)} "
             f"records and {len(FEATURES)} numeric columns, generated with a fixed random seed (42) from three "
             f"overlapping student archetypes with noise. Value ranges: CGPA {mins['CGPA']:.2f} to {maxs['CGPA']:.2f}, "
             f"coding {mins['Coding']:.0f} to {maxs['Coding']:.0f}, attendance {mins['Attendance']:.0f}% to "
             f"{maxs['Attendance']:.0f}%. It has no missing values.</p>")
    gap()
    l, r = st.columns(2)
    with l:
        card("Limitations",
             "<ul><li>The data is simulated, so findings do not describe any real institution.</li>"
             "<li>Clusters show similarity of profiles, not placement outcomes.</li>"
             "<li>K-Means assumes roughly round, similar-sized groups and is sensitive to scaling and outliers.</li>"
             "<li>The silhouette score shows moderate separation, so group borders overlap.</li>"
             "<li>Benchmarks are illustrative, not official criteria.</li>"
             "<li>History lasts only for the current session.</li></ul>")
    with r:
        card("Future enhancements",
             "<ul><li>Use anonymised real institutional data with consent.</li>"
             "<li>Compare K-Means with other clustering methods such as hierarchical clustering or Gaussian mixtures.</li>"
             "<li>Add a database for permanent history.</li>"
             "<li>Add more indicators such as certifications and soft-skill assessments.</li>"
             "<li>Track a student's cluster movement over semesters.</li></ul>")


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    "student_assessment": st.Page(page_assessment, title="Student Assessment", icon=":material/assignment_ind:",
                                  url_path="student_assessment"),
    "cluster_explorer": st.Page(page_clusters, title="Cluster Explorer", icon=":material/bubble_chart:",
                                url_path="cluster_explorer"),
    "model_evaluation": st.Page(page_evaluation, title="Model Evaluation", icon=":material/analytics:",
                                url_path="model_evaluation"),
    "skill_gap": st.Page(page_skill_gap, title="Skill Gap Analysis", icon=":material/track_changes:",
                         url_path="skill_gap"),
    "student_history": st.Page(page_history, title="Student History", icon=":material/history:",
                               url_path="student_history"),
    "about_project": st.Page(page_about, title="About the Project", icon=":material/info:", url_path="about_project"),
}

navigation = st.navigation(list(PAGES.values()))

with st.sidebar:
    html_block(f"""
        <div class="pm-sidepanel">
            <b>Model status</b><br>
            K-Means with k = {K}<br>
            Silhouette score {BUNDLE['silhouette']:.3f}<br>
            {len(DF)} simulated student profiles<br><br>
            Results describe profile similarity. They do not predict placement.
        </div>""")

navigation.run()
