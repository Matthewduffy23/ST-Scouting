# app.py — Polar Percentile Chart for Attackers (no image)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import io
import re

st.set_page_config(page_title="Attacker Polar Profile", layout="wide")
st.title("🎯 Attacker Polar Profile")

# ----------------- Config -----------------
DEFAULT_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Passes to penalty area per 90","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Progressive runs per 90",
]

def clean_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Passes to penalty area per 90", "Passes to penalty area")
    s = s.replace("Shots on target, %", "SoT %")
    s = s.replace("Accurate passes, %", "Pass %")
    s = re.sub(r"\s*per\s*90", "", s, flags=re.I)
    return s

# Attackers position filter you provided
ATT_PREFIXES = ('RWF', 'LWF', 'LAMF', 'RAMF', 'AMF', 'RW, ', 'LW, ')
def is_attacker_position(pos: str) -> bool:
    p = str(pos).strip().upper()
    if p in ('RW', 'LW'):
        return True
    return p.startswith(ATT_PREFIXES)

# ----------------- Data loader -----------------
@st.cache_data(show_spinner=False)
def load_df(csv_name="WORLDJUNE25.csv"):
    p = Path(__file__).with_name(csv_name)
    if p.exists():
        return pd.read_csv(p)
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if not up:
        st.stop()
    return pd.read_csv(up)

df = load_df()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Filters")
    # Minutes / age
    df["Minutes played"] = pd.to_numeric(df.get("Minutes played"), errors="coerce")
    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    min_age, max_age = st.slider("Age", 14, 45, (16, 33))

    # Scope: attackers toggle or manual prefix
    use_attacker_filter = st.checkbox("Attackers only (RW/LW/AMF variants)", True)
    manual_prefix = st.text_input("…or Position startswith (optional)", "")

    # League picker
    leagues_all = sorted(df.get("League", pd.Series([])).dropna().unique().tolist())
    leagues_sel = st.multiselect("Leagues", leagues_all, default=leagues_all)

    # Metrics (ensure present)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics = [m for m in DEFAULT_METRICS if m in df.columns and m in numeric_cols]
    if len(metrics) < 6:
        st.error("Your dataset is missing some of the default attacker metrics.")
        st.stop()

# ----------------- Filter pool -----------------
pool = df[df["League"].isin(leagues_sel)].copy()
if use_attacker_filter:
    pool = pool[pool["Position"].apply(is_attacker_position)]
if manual_prefix.strip():
    pool = pool[pool["Position"].astype(str).str.upper().str.startswith(manual_prefix.strip().upper())]

pool = pool[
    pool["Minutes played"].between(min_minutes, max_minutes) &
    pool["Age"].between(min_age, max_age)
]

# Numeric coercion on needed metrics
for m in metrics:
    pool[m] = pd.to_numeric(pool[m], errors="coerce")
pool = pool.dropna(subset=metrics)

if pool.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

players = sorted(pool["Player"].dropna().unique().tolist())
player = st.selectbox("Select player", players)
prow = pool[pool["Player"] == player].iloc[0]

# ----------------- Percentiles within player's league -----------------
league = prow["League"]
league_pool = pool[pool["League"] == league].copy()

# compute league-wise percentiles for each metric
pct_df = league_pool[metrics].rank(pct=True) * 100.0
player_idx = league_pool.index[league_pool["Player"] == player]
if len(player_idx) == 0:
    st.error("Selected player not found in league pool.")
    st.stop()

percentiles = pct_df.loc[player_idx].mean(axis=0).values.round(0).astype(int).tolist()
labels = [clean_label(m) for m in metrics]

# ----------------- Polar chart renderer (no images) -----------------
def render_percentile_polar_chart(title:str, subtitle:str, metrics_labels:list, percentiles_list:list):
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    norm_vals = [p/100 for p in percentiles_list]
    bar_colors = [cmap(v) for v in norm_vals]

    N = len(metrics_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]  # clockwise
    rotation_shift = np.deg2rad(75) - angles[0]                # start ~1 o'clock
    rotated_angles = (angles + rotation_shift) % (2*np.pi)
    bar_width = 2*np.pi / N

    fig = plt.figure(figsize=(8, 6.5))
    bg = "#e6e6e6"
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.70], polar=True)
    ax.set_facecolor(bg)
    ax.set_rlim(0, 100)

    # Bars + value labels
    for i in range(N):
        ax.bar(rotated_angles[i], percentiles_list[i],
               width=bar_width, color=bar_colors[i],
               edgecolor='black', linewidth=1)
        label_pos = percentiles_list[i] - 10 if percentiles_list[i] >= 15 else percentiles_list[i] * 0.7
        ax.text(rotated_angles[i], label_pos, f"{percentiles_list[i]}",
                ha='center', va='center', fontsize=9, weight='bold', color='white')

    # Outer ring
    outer = plt.Circle((0, 0), 100, transform=ax.transData._b, color='black', fill=False, linewidth=2.4)
    ax.add_artist(outer)

    # Dividers (heavier at cross axes)
    for i in range(N):
        sep_angle = (rotated_angles[i] - bar_width/2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100],
                color='black' if is_cross else '#b0b0b0',
                linewidth=1.8 if is_cross else 1)

    # Metric labels
    label_radius = 125
    for i, lab in enumerate(metrics_labels):
        ax.text(rotated_angles[i], label_radius, lab,
                ha='center', va='center', fontsize=8, weight='bold', color='black')

    # Clean up
    ax.set_xticks([]); ax.set_yticks([]); ax.spines['polar'].set_visible(False); ax.grid(False)

    # Titles
    fig.text(0.05, 0.93, title, fontsize=16, weight='bold', ha='left')
    fig.text(0.05, 0.902, subtitle, fontsize=9, ha='left', color='gray')
    return fig

title = f"{player} — {prow['Team']}"
mins = int(pd.to_numeric(prow.get("Minutes played"), errors="coerce")) if pd.notna(prow.get("Minutes played")) else None
subtitle = f"Percentiles vs {league}" + (f" | {mins:,} mins" if mins is not None else "")

fig = render_percentile_polar_chart(title, subtitle, labels, percentiles)
st.pyplot(fig, use_container_width=True)

# ----------------- Export -----------------
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("⬇️ Download PNG", data=buf.getvalue(),
                   file_name=f"{player.replace(' ','_')}_polar_profile.png",
                   mime="image/png")





