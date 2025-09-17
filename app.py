# app.py — Advanced Scouting Tool (tables, presets, polar & scatter, colored role table,
# notes with style/strengths/weaknesses, comparison radar, similar players, club fit)

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Advanced Scouting Tool", layout="wide")
st.title("🔎 Advanced Scouting Tool")

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.',
    'Germany 4.', 'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.',
    'Israel 2.', 'Italy 1.', 'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.',
    'Kazakhstan 1.', 'Korea 1.', 'Latvia 1.', 'Lithuania 1.', 'Malta 1.',
    'Mexico 1.', 'Moldova 1.', 'Morocco 1.', 'Netherlands 1.', 'Netherlands 2.',
    'North Macedonia 1.', 'Northern Ireland 1.', 'Norway 1.', 'Norway 2.',
    'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.', 'Portugal 1.',
    'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.',
    'Serbia 1.', 'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.',
    'Slovenia 2.', 'South Africa 1.', 'Spain 1.', 'Spain 2.', 'Spain 3.',
    'Sweden 1.', 'Sweden 2.', 'Switzerland 1.', 'Switzerland 2.', 'Tunisia 1.',
    'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.', 'USA 1.', 'USA 2.',
    'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.',
        'Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.',
        'Italy 2.','Czech 1.','Norway 1.'
    },
    "EFL": {'England 2.','England 3.','England 4.'}
}

FEATURES = [
    'Defensive duels per 90', 'Defensive duels won, %',
    'Aerial duels per 90', 'Aerial duels won, %',
    'PAdj Interceptions', 'Non-penalty goals per 90', 'xG per 90',
    'Shots per 90', 'Shots on target, %', 'Goal conversion, %',
    'Crosses per 90', 'Accurate crosses, %', 'Dribbles per 90',
    'Successful dribbles, %', 'Head goals per 90', 'Key passes per 90',
    'Touches in box per 90', 'Progressive runs per 90', 'Accelerations per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90',
    'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
    'Deep completions per 90', 'Smart passes per 90',
]

# Attacker polar chart metrics
POLAR_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Passes to penalty area per 90","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Progressive runs per 90",
]

def clean_attacker_label(s: str) -> str:
    s = s.replace("Non-penalty goals per 90", "Non-Pen Goals")
    s = s.replace("xG per 90", "xG").replace("xA per 90", "xA")
    s = s.replace("Shots per 90", "Shots")
    s = s.replace("Passes per 90", "Passes")
    s = s.replace("Touches in box per 90", "Touches in box")
    s = s.replace("Aerial duels per 90", "Aerial duels")
    s = s.replace("Progressive runs per 90", "Progressive runs")
    s = s.replace("Passes to penalty area per 90", "Passes to Pen area")
    s = s.replace("Accurate passes, %", "Pass %")
    return s

ROLES = {
    'Target Man CF': {
        'desc': "Aerial outlet, duel dominance, occupy CBs, threaten crosses & second balls.",
        'metrics': { 'Aerial duels per 90': 3, 'Aerial duels won, %': 4 }
    },
    'Goal Threat CF': {
        'desc': "High shot & xG volume, box presence, consistent SoT and finishing.",
        'metrics': {
            'Non-penalty goals per 90': 3, 'Shots per 90': 1.5,
            'xG per 90': 3, 'Touches in box per 90': 1, 'Shots on target, %': 0.5
        }
    },
    'Link-Up CF': {
        'desc': "Combine & create; link play; progress & deliver to the penalty area.",
        'metrics': {
            'Passes per 90': 2, 'Passes to penalty area per 90': 1.5,
            'Deep completions per 90': 1, 'Smart passes per 90': 1.5,
            'Accurate passes, %': 1.5, 'Key passes per 90': 1,
            'Dribbles per 90': 2, 'Successful dribbles, %': 1,
            'Progressive runs per 90': 2, 'xA per 90': 3
        }
    },
    'All in': {
        'desc': "Blend of creation + scoring; balanced all-round attacking profile.",
        'metrics': { 'xA per 90': 2, 'Dribbles per 90': 2, 'xG per 90': 3, 'Non-penalty goals per 90': 3 }
    }
}

LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 9.':12.00,'Andorra 1.':10.00,'Estonia 1.':23.00,
    'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}

REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Contract expires","Goals"}

# ------- Style / strengths / weaknesses mapping (labels WITHOUT ordinal suffix) ------
# threshold: >=85 strong, <=35 weak; style uses "volume/usage" readings on 60+ or below 40 for some
STYLE_MAP = {
    'Defensive duels per 90': {'style': 'High Work Rate', 'strong': 'Defensive Duels', 'weak': 'Defensive Duels'},
    'Aerial duels per 90': {'style': 'Outlet Presence', 'strong': 'Aerial Presence', 'weak': 'Aerial Presence'},
    'Aerial duels won, %': {'style': None, 'strong': 'Aerial Duels', 'weak': 'Aerial Duels'},
    'Non-penalty goals per 90': {'style': None, 'strong': 'Scoring Goals', 'weak': 'Scoring Goals'},
    'xG per 90': {'style': 'Attacking Positioning', 'strong': 'Chance Quality (xG)', 'weak': 'Chance Quality (xG)'},
    'Shots per 90': {'style': 'Shot Volume', 'strong': None, 'weak': None},
    'Goal conversion, %': {'style': None, 'strong': 'Finishing', 'weak': 'Finishing'},
    'Crosses per 90': {'style': 'Crossing Volume', 'strong': None, 'weak': None},
    'Accurate crosses, %': {'style': None, 'strong': 'Crossing', 'weak': 'Crossing'},
    'Dribbles per 90': {'style': '1v1 Dribbler', 'strong': None, 'weak': None},
    'Successful dribbles, %': {'style': None, 'strong': 'Dribbling', 'weak': 'Dribbling'},
    'Touches in box per 90': {'style': 'Busy in Box', 'strong': None, 'weak': None},
    'Progressive runs per 90': {'style': 'Ball Carrier', 'strong': 'Progressive Runs', 'weak': 'Progressive Runs'},
    'Passes per 90': {'style': 'Busy in Build-up', 'strong': None, 'weak': None},
    'Accurate passes, %': {'style': None, 'strong': 'Retention', 'weak': 'Retention'},
    'xA per 90': {'style': 'Chance Creator', 'strong': 'Creating Chances (xA)', 'weak': 'Creating Chances (xA)'},
    'Passes to penalty area per 90': {'style': 'Facilitator', 'strong': 'Passes to Penalty Area', 'weak': 'Passes to Penalty Area'},
    'Deep completions per 90': {'style': 'Value-Adding Passer', 'strong': None, 'weak': None},
    'Smart passes per 90': {'style': 'Line-Breaking Passer', 'strong': None, 'weak': None},
}

# ----------------- DATA LOADER -----------------
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

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filters")

    # Preset toggles -> seed multiselect
    c1, c2, c3 = st.columns([1,1,1])
    use_top5  = c1.checkbox("Top-5", value=False)
    use_top20 = c2.checkbox("Top-20", value=False)
    use_efl   = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues (add or prune the presets)", leagues_avail, default=default_leagues)

    # Numerics for bounds
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 33))

    pos_text = st.text_input("Position startswith", "CF")

    # Defaults OFF; beta default 0.4
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.4, 0.05,
                     help="0 = ignore league strength; 1 = only league strength")

    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, max_m))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, mv_cap), step=100_000)
    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0,
                                     value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:],
                                 default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
    min_pct = st.slider("Minimum percentile (0–100)", 0, 100, 60)

    top_n = st.number_input("Top N per table", 5, 200, 50, 5)
    round_to = st.selectbox("Round output percentiles to", [0, 1], index=0)

# ----------------- VALIDATION -----------------
missing = [c for c in REQUIRED_BASE if c not in df.columns]
if missing:
    st.error(f"Dataset missing required base columns: {missing}")
    st.stop()

missing_feats = [c for c in FEATURES if c not in df.columns]
if missing_feats:
    st.error(f"Dataset missing required feature columns: {missing_feats}")
    st.stop()

# ----------------- FILTER POOL FOR TABLES -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).str.startswith(tuple([pos_text]))]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
df_f = df_f.dropna(subset=FEATURES)

df_f["Contract expires"] = pd.to_datetime(df_f["Contract expires"], errors="coerce")
if apply_contract:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]

df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- PERCENTILES (per league) for tables -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)
for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- ROLE SCORING for tables -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100
    if league_weighting:
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        return (1 - beta) * player_score + beta * league_scaled
    return player_score

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(df_f, role_def["metrics"], beta=beta, league_weighting=use_league_weighting)

# ----------------- MINIMUM PERFORMANCE THRESHOLDS -----------------
if enable_min_perf and sel_metrics:
    keep_mask = np.ones(len(df_f), dtype=bool)
    for m in sel_metrics:
        pct_col = f"{m} Percentile"
        if pct_col in df_f.columns:
            keep_mask &= (df_f[pct_col] >= min_pct)
    df_f = df_f[keep_mask]
    if df_f.empty:
        st.warning("No players meet the minimum performance thresholds. Loosen thresholds.")
        st.stop()

# ----------------- HELPERS -----------------
def fmt_cols(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[score_col] = out[score_col].round(round_to).astype(int if round_to == 0 else float)
    cols = ["Player","Team","League","Age","Contract expires","League Strength", score_col]
    return out[cols]

def top_table(df_in: pd.DataFrame, role: str, head_n: int) -> pd.DataFrame:
    col = f"{role} Score"
    ranked = df_in.dropna(subset=[col]).sort_values(col, ascending=False)
    ranked = fmt_cols(ranked, col).head(head_n).reset_index(drop=True)
    ranked.index = np.arange(1, len(ranked)+1)
    return ranked

def filtered_view(df_in: pd.DataFrame, *, age_max=None, contract_year=None, value_max=None):
    t = df_in.copy()
    if age_max is not None:
        t = t[t["Age"] <= age_max]
    if contract_year is not None:
        t = t[t["Contract expires"].dt.year <= contract_year]
    if value_max is not None:
        t = t[t["Market value"] <= value_max]
    return t

# ----------------- TABS -----------------
tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)"])

for role, role_def in ROLES.items():
    with tabs[0]:
        st.subheader(f"{role} — Overall Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[1]:
        u23_cutoff = st.number_input(f"{role} — U23 cutoff", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        st.subheader(f"{role} — U{u23_cutoff} Top {int(top_n)}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, age_max=u23_cutoff), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[2]:
        exp_year = st.number_input(f"{role} — Expiring by year", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        st.subheader(f"{role} — Contracts expiring ≤ {exp_year}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, contract_year=exp_year), role, int(top_n)), use_container_width=True)
        st.divider()
    with tabs[3]:
        v_max = st.number_input(f"{role} — Max value (€)", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        st.subheader(f"{role} — Value band ≤ €{v_max:,.0f}")
        st.caption(role_def.get("desc", ""))
        st.dataframe(top_table(filtered_view(df_f, value_max=v_max), role, int(top_n)), use_container_width=True)
        st.divider()

# ----------------- SINGLE PLAYER ROLE PROFILE (custom pool) -----------------
st.subheader("🎯 Single Player Role Profile")
player_name = st.selectbox("Choose player", sorted(df_f["Player"].unique()))
player_row = df_f[df_f["Player"] == player_name].head(1)

# Controls for comparison pool (defaults: player's league only)
if not player_row.empty:
    default_league = [player_row["League"].iloc[0]]
else:
    default_league = []
st.caption("Percentiles & chart are computed against the pool below (defaults to player's league).")
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    leagues_pool = c1.multiselect("Comparison leagues", sorted(df["League"].dropna().unique()), default=default_league)
    min_minutes_pool, max_minutes_pool = c2.slider("Pool minutes", 0, 5000, (500, 5000))
    # Pool age default 16–40
    age_min_pool, age_max_pool = c3.slider("Pool age", 14, 45, (16, 40))
    same_pos = st.checkbox("Limit pool to current position prefix", value=True)
    c4, c5 = st.columns([1.2, 2])
    # Player role league weighting: default OFF, beta default 0.4
    use_player_league_weight = c4.checkbox("Weight player role scores by league", value=False)
    beta_player = c5.slider("Player role beta (league vs. metrics)", 0.0, 1.0, 0.4, 0.05)

def build_pool_df():
    if not leagues_pool:
        return pd.DataFrame([], columns=df.columns)
    pool = df[df["League"].isin(leagues_pool)].copy()
    pool["Minutes played"] = pd.to_numeric(pool["Minutes played"], errors="coerce")
    pool["Age"] = pd.to_numeric(pool["Age"], errors="coerce")
    pool = pool[pool["Minutes played"].between(min_minutes_pool, max_minutes_pool)]
    pool = pool[pool["Age"].between(age_min_pool, age_max_pool)]
    if same_pos and not player_row.empty:
        pref = str(player_row["Position"].iloc[0])[:2]  # e.g., CF
        pool = pool[pool["Position"].astype(str).str.startswith(pref)]
    pool = pool.dropna(subset=POLAR_METRICS)
    return pool

def percentiles_for_player_in_pool(pool_df: pd.DataFrame, ply_row: pd.Series) -> dict:
    """Return {metric: percentile 0..100} computed across the pool (combined), not per league."""
    if pool_df.empty:
        return {}
    pct_map = {}
    for m in POLAR_METRICS:
        if m not in pool_df.columns or pd.isna(ply_row[m]):
            continue
        series = pd.to_numeric(pool_df[m], errors="coerce").dropna()
        if series.empty:
            continue
        rank = (series < float(ply_row[m])).mean() * 100.0
        eq_share = (series == float(ply_row[m])).mean() * 100.0
        pct_map[m] = min(100.0, rank + 0.5 * eq_share)
    return pct_map

def player_role_scores_from_pct(pct_map: dict, *, player_league_strength: float, use_weight: bool, beta: float) -> dict:
    out = {}
    for role, rd in ROLES.items():
        weights = rd["metrics"]
        total = sum(weights.values()) or 1.0
        metric_score = sum((pct_map.get(m, np.nan) if m in pct_map else 0) * w for m, w in weights.items())
        # guard NaNs
        metric_score = metric_score / total if total > 0 else np.nan
        if use_weight:
            league_scaled = (player_league_strength / 100.0) * 100.0
            out[role] = (1 - beta) * metric_score + beta * league_scaled
        else:
            out[role] = metric_score
    return out

# Gradient helper for role table (0->100 = red->gold->green)
def score_to_color(v: float) -> str:
    if pd.isna(v): return "background-color: #ffffff"
    if v <= 50:
        r1,g1,b1 = (190,42,62); r2,g2,b2 = (244,209,102); t = v/50
    else:
        r1,g1,b1 = (244,209,102); r2,g2,b2 = (34,197,94); t = (v-50)/50
    r = int(r1 + (r2-r1)*t); g = int(g1 + (g2-g1)*t); b = int(b1 + (b2-b1)*t)
    return f"background-color: rgb({r},{g},{b})"

# Polar chart
def plot_attacker_polar_chart(labels, vals):
    N = len(labels)
    color_scale = ["#be2a3e", "#e25f48", "#f88f4d", "#f4d166", "#90b960", "#4b9b5f", "#22763f"]
    cmap = LinearSegmentedColormap.from_list("custom_scale", color_scale)
    bar_colors = [cmap(v/100.0) for v in vals]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False)[::-1]
    rotation_shift = np.deg2rad(75) - angles[0]
    ang = (angles + rotation_shift) % (2*np.pi)
    width = 2*np.pi / N

    fig = plt.figure(figsize=(8.2, 6.6), dpi=180)
    fig.patch.set_facecolor('#f3f4f6')
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.74], polar=True)
    ax.set_facecolor('#f3f4f6')
    ax.set_rlim(0, 100)

    for i in range(N):
        ax.bar(ang[i], vals[i], width=width, color=bar_colors[i], edgecolor='black', linewidth=1.0, zorder=3)
        label_pos = max(12, vals[i] * 0.75)
        ax.text(ang[i], label_pos, f"{int(round(vals[i]))}", ha='center', va='center',
                fontsize=9, weight='bold', color='white', zorder=4)

    outer = plt.Circle((0, 0), 100, transform=ax.transData._b, color='black', fill=False, linewidth=2.2, zorder=5)
    ax.add_artist(outer)
    for i in range(N):
        sep_angle = (ang[i] - width/2) % (2*np.pi)
        is_cross = any(np.isclose(sep_angle, a, atol=0.01) for a in [0, np.pi/2, np.pi, 3*np.pi/2])
        ax.plot([sep_angle, sep_angle], [0, 100], color='black' if is_cross else '#b0b0b0',
                linewidth=1.6 if is_cross else 1.0, zorder=2)

    label_r = 120
    for i, lab in enumerate(labels):
        ax.text(ang[i], label_r, lab, ha='center', va='center', fontsize=8.5, weight='bold', color='#111827', zorder=6)

    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False); ax.grid(False)
    return fig

# --- Player UI ---
if player_row.empty:
    st.info("Pick a player above.")
else:
    meta = player_row[["Team","League","Age","Contract expires","League Strength","Market value"]].iloc[0]
    st.caption(
        f"**{player_name}** — {meta['Team']} • {meta['League']} • Age {int(meta['Age'])} • "
        f"Contract: {pd.to_datetime(meta['Contract expires']).date() if pd.notna(meta['Contract expires']) else 'N/A'} • "
        f"League Strength {meta['League Strength']:.1f} • Value €{meta['Market value']:,.0f}"
    )

    # Build pool & compute player percentiles within that pool
    pool_df = build_pool_df()
    if pool_df.empty:
        st.warning("Comparison pool is empty. Add at least one league.")
    else:
        ply = player_row.iloc[0]
        pct_map = percentiles_for_player_in_pool(pool_df, ply)

        # Role scores based on pool percentiles (with optional league weighting for the player)
        player_ls = float(LEAGUE_STRENGTHS.get(str(ply["League"]), 50.0))
        role_scores = player_role_scores_from_pct(
            pct_map,
            player_league_strength=player_ls,
            use_weight=use_player_league_weight,
            beta=beta_player
        )

        # Role table with colors
        rows = [{"Role": r, "Percentile": role_scores.get(r, np.nan)} for r in ROLES.keys()]
        role_df = pd.DataFrame(rows).set_index("Role")
        styled = role_df.style.applymap(lambda x: score_to_color(float(x)) if pd.notna(x) else "background-color:#fff",
                                        subset=["Percentile"]) \
                               .format({"Percentile": lambda x: f"{int(round(x))}" if pd.notna(x) else "—"})
        st.dataframe(styled, use_container_width=True)

        # Polar chart (pool percentiles)
        labels = [clean_attacker_label(m) for m in POLAR_METRICS if m in pct_map]
        vals   = [pct_map[m] for m in POLAR_METRICS if m in pct_map]
        if vals:
            fig = plot_attacker_polar_chart(labels, vals)
            team = str(ply["Team"]); league = str(ply["League"])
            fig.text(0.06, 0.94, f"{player_name} — Performance (pool size: {len(pool_df):,})",
                     fontsize=16, weight='bold', ha='left', color='#111827')
            fig.text(0.06, 0.915, f"Against selected pool • Team: {team} • Native league: {league}",
                     fontsize=9, ha='left', color='#6b7280')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough metrics to draw the polar chart.")

# ----------------- SCATTER: NPG/90 (X) vs xG/90 (Y) on same pool -----------------
st.subheader("Scatter: Non-penalty goals per 90 vs xG per 90 (adjustable pool above)")
x_col = "Non-penalty goals per 90"
y_col = "xG per 90"
if 'pool_df' in locals() and not player_row.empty and not pool_df.empty and (x_col in pool_df.columns) and (y_col in pool_df.columns):
    scat = pool_df[[x_col, y_col, "Player"]].dropna().copy()

    # Medians
    x_med = np.median(pd.to_numeric(scat[x_col], errors="coerce"))
    y_med = np.median(pd.to_numeric(scat[y_col], errors="coerce"))

    fig2, ax2 = plt.subplots(figsize=(10.5, 7.5), dpi=160)
    fig2.patch.set_facecolor("#f3f4f6")
    ax2.set_facecolor("#f3f4f6")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.6, color="#e5e7eb")

    # all points — black, slightly larger circles
    ax2.scatter(scat[x_col], scat[y_col], s=36, c="black", alpha=0.8,
                edgecolors="white", linewidths=0.5)

    # highlight selected player
    if player_name in scat["Player"].values:
        pr = scat[scat["Player"] == player_name].iloc[0]
        ax2.scatter([pr[x_col]], [pr[y_col]], s=160, c="red", edgecolors="white", linewidths=1.2, zorder=5)
        ax2.annotate(player_name, (pr[x_col], pr[y_col]), xytext=(10,10), textcoords="offset points",
                     color="red", fontsize=10, weight="bold")

    # medians
    ax2.axvline(x_med, color="#9ca3af", linestyle="--", linewidth=1.2, zorder=0)
    ax2.axhline(y_med, color="#9ca3af", linestyle="--", linewidth=1.2, zorder=0)

    # limits & labels
    ax2.set_xlim(0, max(1.0, float(scat[x_col].max()) * 1.05))
    ax2.set_ylim(0, max(1.0, float(scat[y_col].max()) * 1.05))
    ax2.set_xlabel("Non-penalty goals per 90", fontweight="bold")
    ax2.set_ylabel("xG per 90", fontweight="bold")
    ax2.set_title("Shot output vs shot quality (medians shown; grey background)", fontweight="bold")

    st.pyplot(fig2, use_container_width=True)
else:
    st.info("Add at least one league to the comparison pool (and ensure required metrics exist).")

# ----------------- AI-STYLE SUMMARY NOTES (no API) -----------------
st.subheader("📝 Notes")
if not player_row.empty:
    ply = player_row.iloc[0]
    # Use pool percentiles if available, else table-based league percentiles
    source_map = {}
    if 'pct_map' in locals() and pct_map:
        source_map = pct_map
    else:
        for m in POLAR_METRICS:
            col = f"{m} Percentile"
            if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                source_map[m] = float(player_row[col].iloc[0])

    # Strengths / Weaknesses lists (labels only, no "99th")
    strengths, weaknesses, styles = [], [], []
    for metric, conf in STYLE_MAP.items():
        # derive percentile source: pool if available else per-league (if present)
        val = None
        if metric in source_map:
            val = source_map[metric]
        else:
            col = f"{metric} Percentile"
            if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                val = float(player_row[col].iloc[0])
        if val is None:
            continue

        # style
        if conf.get('style'):
            if val >= 60:
                styles.append(conf['style'])
            elif val <= 40 and conf['style'] in ['Busy in Build-up', 'Shot Volume', 'Crossing Volume', 'Outlet Presence', 'High Work Rate', 'Busy in Box']:
                # optional negative style cues; usually we only list positive styles
                pass

        # strengths/weaknesses (no ordinal in label)
        if conf.get('strong') and val >= 85:
            strengths.append(conf['strong'])
        if conf.get('weak') and val <= 35:
            weaknesses.append(conf['weak'])

    # Best role from the first three titles (Target Man CF, Goal Threat CF, Link-Up CF)
    first_three = ['Target Man CF','Goal Threat CF','Link-Up CF']
    best_role = None
    best_val = -1
    for r in first_three:
        col = f"{r} Score"
        if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
            v = float(player_row[col].iloc[0])
            if v > best_val:
                best_val = v
                best_role = r

    # Render
    st.markdown(
        f"**Profile:** {player_name} — {ply['Team']} ({ply['League']}), age {int(ply['Age'])}, minutes {int(ply['Minutes played'])}."
    )
    if best_role is not None:
        st.markdown(f"**Best role:** {best_role}")

    def chips(items, color_hex):
        if not items: return "_None identified._"
        spans = [f"<span style='background:{color_hex};color:#111;padding:2px 6px;border-radius:10px;margin-right:6px;display:inline-block'>{name}</span>"
                 for name in sorted(set(items))[:10]]
        return " ".join(spans)

    st.markdown("**Style:**", unsafe_allow_html=True)
    st.markdown(chips(styles, "#e5e7eb"), unsafe_allow_html=True)          # grey tags

    st.markdown("**Strengths:**", unsafe_allow_html=True)
    st.markdown(chips(strengths, "#a7f3d0"), unsafe_allow_html=True)       # light green

    st.markdown("**Weaknesses / growth areas:**", unsafe_allow_html=True)
    st.markdown(chips(weaknesses, "#fecaca"), unsafe_allow_html=True)      # light red
else:
    st.caption("Pick a player above to generate notes.")

# ===============================
# BELOW: add-ons under player notes
# ===============================

# ------------------------------
# 1) SB-style comparison radar
# ------------------------------
st.subheader("🔁 Compare two players — SB-style radar")

if player_row.empty:
    st.info("Pick a player above to enable comparison.")
else:
    # Build pool for radar (reuse comparison pool; if empty, fallback to df_f)
    _pool_for_radar = build_pool_df()
    if _pool_for_radar.empty:
        _pool_for_radar = df_f.copy()
    # same position family as selected player
    _pos_pref = str(player_row["Position"].iloc[0])[:2]
    _pool_for_radar = _pool_for_radar[_pool_for_radar["Position"].astype(str).str.startswith(_pos_pref)]

    # Opponent chooser
    options = sorted(_pool_for_radar["Player"].dropna().unique().tolist())
    if player_name in options:
        options.remove(player_name)
    pB = st.selectbox("Compare with (Player B — blue)", options) if options else None

    # Radar metrics selector
    default_metrics = [m for m in POLAR_METRICS if m in df.columns]
    metrics_radar = st.multiselect("Radar metrics", default_metrics, default_metrics)
    sort_by_gap = st.checkbox("Sort axes by biggest gap", False)
    show_avg    = st.checkbox("Show pool average (thin line)", True)

    if pB and len(metrics_radar) >= 5:
        pool_r = _pool_for_radar[metrics_radar].rank(pct=True) * 100.0

        def _pcts_for(name):
            idx = _pool_for_radar[_pool_for_radar["Player"] == name].index
            return pool_r.loc[idx].mean(axis=0).values if len(idx) else np.full(len(metrics_radar), np.nan)

        A_r = _pcts_for(player_name)
        B_r = _pcts_for(pB)

        if sort_by_gap:
            order = np.argsort(-np.abs(A_r - B_r))
            metrics_radar = [metrics_radar[i] for i in order]
            A_r = A_r[order]
            B_r = B_r[order]

        # small SB-look radar
        def _draw_sb_radar(labels, A_r, B_r, show_avg=True):
            labels = [clean_attacker_label(x) for x in labels]
            N = len(labels)
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            theta_c = np.r_[theta, theta[:1]]
            Ar = np.r_[A_r, A_r[:1]]
            Br = np.r_[B_r, B_r[:1]]

            fig = plt.figure(figsize=(11.5, 7.2), dpi=210)
            ax = plt.subplot(111, polar=True)
            fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
            ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)

            ax.set_xticks(theta); ax.set_xticklabels(labels, fontsize=10, color="#0f172a", fontweight=600)
            ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values(): s.set_visible(False)

            # alternating annuli
            ring_r = np.linspace(0, 100, 11)
            for i in range(len(ring_r)-1):
                r0, r1 = ring_r[i], ring_r[i+1]
                ax.add_artist(Wedge((0,0), r1, 0, 360, width=(r1-r0),
                                    transform=ax.transData._b,
                                    facecolor=("#ffffff" if i%2==0 else "#e5e7eb"),
                                    edgecolor="none", zorder=0.5))
            # faint circles
            t = np.linspace(0, 2*np.pi, 361)
            for r in ring_r:
                ax.plot(t, np.full_like(t, r), color="#d1d5db", lw=1.0, zorder=0.6)

            if show_avg:
                Avg = np.full(N, 50.0)
                ax.plot(theta_c, np.r_[Avg, Avg[:1]], lw=1.3, color="#94a3b8", ls="--", zorder=2)

            ax.plot(theta_c, Ar, color="#c81e1e", lw=2.2, zorder=3)
            ax.fill(theta_c, Ar, color=(200/255,30/255,30/255,0.55), zorder=2.4)

            ax.plot(theta_c, Br, color="#1d4ed8", lw=2.2, zorder=3)
            ax.fill(theta_c, Br, color=(29/255,78/255,216/255,0.55), zorder=2.4)

            ax.set_rlim(0, 104)
            metaA = player_row.iloc[0]
            metaB = _pool_for_radar[_pool_for_radar["Player"] == pB].iloc[0]
            fig.text(0.12, 0.96, f"{player_name}", color="#c81e1e", fontsize=20, weight="bold", ha="left")
            fig.text(0.12, 0.935, f"{metaA['Team']} — {metaA['League']}", color="#c81e1e", fontsize=10, ha="left")
            fig.text(0.88, 0.96,  f"{pB}", color="#1d4ed8", fontsize=20, weight="bold", ha="right")
            fig.text(0.88, 0.935, f"{metaB['Team']} — {metaB['League']}", color="#1d4ed8", fontsize=10, ha="right")
            return fig

        st.pyplot(_draw_sb_radar(metrics_radar, A_r, B_r, show_avg=show_avg), use_container_width=True)
    else:
        st.caption("Pick at least 5 metrics and a comparison player.")

st.divider()

# ------------------------------
# 2) Similar players (within current comparison pool)
# ------------------------------
st.subheader("🧭 Similar players (within adjustable pool)")

if player_row.empty:
    st.info("Pick a player above to enable similarity search.")
else:
    # Similarity controls
    with st.expander("Similarity settings"):
        percentile_weight = st.slider("Percentile distance weight", 0.0, 1.0, 0.7, 0.05)
        league_weight     = st.slider("League adjustment weight", 0.0, 1.0, 0.2, 0.05)
        st.caption("Combine percentile-based distance (within pool) with standardized value distance.")
        st.markdown("**Advanced feature weights**")
        # default weights
        weight_factors = {
            'Passes per 90': 3, 'Dribbles per 90': 3, 'Non-penalty goals per 90': 3,
            'Aerial duels won, %': 2, 'Aerial duels per 90': 3,
            'xA per 90': 2, 'xG per 90': 3, 'Touches in box per 90': 2,
        }
        wf = weight_factors.copy()
        colA, colB, colC, colD = st.columns(4)
        wf['Passes per 90']            = colA.slider("Passes per 90", 1, 5, wf['Passes per 90'])
        wf['Dribbles per 90']          = colA.slider("Dribbles per 90", 1, 5, wf['Dribbles per 90'])
        wf['Non-penalty goals per 90'] = colB.slider("Non-penalty goals per 90", 1, 5, wf['Non-penalty goals per 90'])
        wf['Aerial duels per 90']      = colB.slider("Aerial duels per 90", 1, 5, wf['Aerial duels per 90'])
        wf['Aerial duels won, %']      = colC.slider("Aerial duels won, %", 1, 5, wf['Aerial duels won, %'])
        wf['xG per 90']                = colC.slider("xG per 90", 1, 5, wf['xG per 90'])
        wf['xA per 90']                = colD.slider("xA per 90", 1, 5, wf['xA per 90'])
        wf['Touches in box per 90']    = colD.slider("Touches in box per 90", 1, 5, wf['Touches in box per 90'])
        show_k = st.number_input("Show top K similar", 5, 200, 50, 5)

    pool = build_pool_df()
    if pool.empty:
        st.info("Your comparison pool is empty. Add leagues above.")
    else:
        # ensure selected player exists in global df for standardized features
        target_full = df[df["Player"] == player_name].head(1)
        if target_full.empty:
            st.warning("Selected player not found in dataset.")
        else:
            features_sim = [
                'Defensive duels per 90', 'Aerial duels per 90', 'Aerial duels won, %',
                'Non-penalty goals per 90', 'xG per 90', 'Shots per 90', 'Shots on target, %',
                'Crosses per 90',  'Dribbles per 90', 'Successful dribbles, %',
                'Offensive duels per 90', 'Touches in box per 90', 'Progressive runs per 90',
                'Passes per 90', 'Accurate passes, %', 'xA per 90', 'Smart passes per 90',
                'Passes to final third per 90', 'Passes to penalty area per 90',
                'Deep completions per 90'
            ]
            req = {'Player','Team','League','Age','Minutes played','Market value', *features_sim}
            if any(c not in df.columns for c in req):
                st.warning("Dataset is missing columns required for similarity block.")
            else:
                cand = pool.dropna(subset=features_sim).copy()
                cand = cand[cand['Player'] != player_name]
                if cand.empty:
                    st.info("No candidates after filtering.")
                else:
                    # percentile ranks within each candidate's league (inside pool leagues)
                    perc = cand.groupby('League')[features_sim].rank(pct=True).values
                    # target percentiles over its league (use full df for stability)
                    target_perc = (
                        df.groupby('League')[features_sim]
                        .rank(pct=True)
                        .loc[df['Player'] == player_name]
                        .values
                    )
                    # standardize over candidates
                    scaler = StandardScaler()
                    std_c = scaler.fit_transform(cand[features_sim])
                    target_std = scaler.transform(target_full[features_sim])

                    weights = np.array([wf.get(f, 1) for f in features_sim], dtype=float)
                    d_pct = np.linalg.norm((perc - target_perc) * weights, axis=1)
                    d_std = np.linalg.norm((std_c - target_std) * weights, axis=1)

                    combined = percentile_weight * d_pct + (1 - percentile_weight) * d_std

                    # normalize to 0..100 similarity
                    norm = (combined - combined.min()) / (combined.ptp() if combined.ptp() != 0 else 1.0)
                    sim = (1 - norm) * 100

                    out = cand.copy()
                    out['Similarity'] = sim.round(2)

                    # league adjustment against player's native league
                    target_ls = LEAGUE_STRENGTHS.get(str(target_full['League'].iloc[0]), 50.0)
                    out['League strength'] = out['League'].map(LEAGUE_STRENGTHS).fillna(0.0)
                    ratio = (out['League strength'] / target_ls).clip(0.5, 1.2)
                    out['Adjusted Similarity'] = out['Similarity'] * (1 - league_weight) + out['Similarity'] * ratio * league_weight

                    out = out.sort_values('Adjusted Similarity', ascending=False).reset_index(drop=True)
                    out.insert(0, 'Rank', np.arange(1, len(out)+1))
                    cols_to_show = ['Rank','Player','Team','League','Age','Minutes played','Market value','League strength','Similarity','Adjusted Similarity']
                    cols_to_show = [c for c in cols_to_show if c in out.columns]
                    st.dataframe(out[cols_to_show].head(int(show_k)), use_container_width=True)

st.divider()

# ------------------------------
# 3) Club Fit (team similarity to player)
# ------------------------------
st.subheader("🏟️ Club Fit (vs selected player)")
if player_row.empty:
    st.info("Pick a player above to enable club fit.")
else:
    with st.expander("Club fit settings"):
        # Candidate leagues (preset + extras)
        leagues_available = sorted(set(INCLUDED_LEAGUES) | set(df.get('League', pd.Series([])).dropna().unique()))
        if 'clubfit_leagues' not in st.session_state:
            st.session_state.clubfit_leagues = INCLUDED_LEAGUES.copy()
        preset_name_cf = st.selectbox("Candidate pool preset", list(PRESET_LEAGUES.keys()), index=1)
        if st.button("Apply preset (club fit)"):
            st.session_state.clubfit_leagues = sorted(PRESET_LEAGUES[preset_name_cf])

        extra_cf = st.multiselect("Extra leagues to add", leagues_available, default=[])
        leagues_cf = sorted(set(st.session_state.clubfit_leagues) | set(extra_cf))
        st.caption(f"Club-fit candidate leagues: **{len(leagues_cf)}**")

        pos_scope_cf = st.text_input("Position startswith (club fit)", str(player_row["Position"].iloc[0])[:2])
        # filters for building team profiles
        max_minutes_in_data = int(pd.to_numeric(df.get('Minutes played', pd.Series([0])), errors='coerce').fillna(0).max())
        min_minutes_cf, max_minutes_cf = st.slider("Minutes filter (club fit)", 0, max(1000, max_minutes_in_data),
                                                   (500, max(1000, max_minutes_in_data)))
        age_series = pd.to_numeric(df.get('Age', pd.Series([16, 45])), errors='coerce')
        age_min_data = int(np.nanmin(age_series)) if age_series.notna().any() else 14
        age_max_data = int(np.nanmax(age_series)) if age_series.notna().any() else 45
        min_age_cf, max_age_cf = st.slider("Age filter (club fit)", age_min_data, age_max_data, (16, 33))

        min_strength_cf, max_strength_cf = st.slider("League strength (club fit)", 0, 101, (0, 101))
        league_weight_cf = st.slider("League weight (club fit)", 0.0, 1.0, 0.4, 0.05)
        market_weight_cf = st.slider("Market value weight (club fit)", 0.0, 1.0, 0.2, 0.05)

        manual_mv = st.number_input("Target market value override (€)", min_value=0, value=0, step=100000)

        st.markdown("**Advanced feature weights (club fit)**")
        # default club-fit weights
        default_weight_factors = {
            'Passes per 90': 2, 'Accurate passes, %': 2, 'Dribbles per 90': 2,
            'Non-penalty goals per 90': 2, 'Shots per 90': 2, 'Successful dribbles, %': 2,
            'Aerial duels won, %': 2, 'xA per 90': 2, 'xG per 90': 2, 'Touches in box per 90': 2,
        }
        weights_ui = {}
        for f in FEATURES:
            weights_ui[f] = st.slider(f"• {f}", 0, 5, int(default_weight_factors.get(f, 1)))
        top_n_cf = st.number_input("Show top N teams", 5, 100, 20, 5)

    # ---- Compute club fit ----
    # Candidate players for team profiles
    df_candidates = df[df['League'].isin(leagues_cf)].copy()
    df_candidates = df_candidates[df_candidates['Position'].astype(str).str.startswith(tuple([pos_scope_cf]))]

    df_candidates['Minutes played'] = pd.to_numeric(df_candidates['Minutes played'], errors='coerce')
    df_candidates['Age'] = pd.to_numeric(df_candidates['Age'], errors='coerce')
    df_candidates['Market value'] = pd.to_numeric(df_candidates['Market value'], errors='coerce')

    df_candidates = df_candidates[
        df_candidates['Minutes played'].between(min_minutes_cf, max_minutes_cf, inclusive='both')
    ]
    df_candidates = df_candidates[
        df_candidates['Age'].between(min_age_cf, max_age_cf, inclusive='both')
    ]
    df_candidates = df_candidates.dropna(subset=FEATURES)

    if df_candidates.empty:
        st.warning("No candidate players after club fit filters.")
    else:
        # Target row (use global df_f to preserve typed pos/filters for player)
        trow = player_row.iloc[0]
        target_vector = trow[FEATURES].values
        target_ls = LEAGUE_STRENGTHS.get(trow['League'], 1.0)

        mv_t = trow.get('Market value', np.nan)
        if manual_mv and manual_mv > 0:
            target_market_value = float(manual_mv)
        else:
            target_market_value = float(mv_t) if pd.notna(mv_t) and mv_t > 0 else 2_000_000.0

        # Team profiles
        club_profiles = df_candidates.groupby(['Team'])[FEATURES].mean().reset_index()
        team_league = df_candidates.groupby('Team')['League'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        team_market = df_candidates.groupby('Team')['Market value'].mean()

        club_profiles['League'] = club_profiles['Team'].map(team_league)
        club_profiles['Avg Team Market Value'] = club_profiles['Team'].map(team_market)
        club_profiles = club_profiles.dropna(subset=['Avg Team Market Value'])

        # Similarity (feature scaling + weights)
        weights_vec = np.array([weights_ui.get(f, 1) for f in FEATURES], dtype=float)
        scaler_cf = StandardScaler()
        scaled_team = scaler_cf.fit_transform(club_profiles[FEATURES])
        target_scaled = scaler_cf.transform([target_vector])[0]

        dist = np.linalg.norm((scaled_team - target_scaled) * weights_vec, axis=1)
        rng = dist.max() - dist.min()
        base_fit = (1 - (dist - dist.min()) / (rng if rng > 0 else 1)) * 100
        club_profiles['Club Fit %'] = base_fit.round(2)

        # League strength adjustment
        club_profiles['League strength'] = club_profiles['League'].map(LEAGUE_STRENGTHS).fillna(0.0)
        club_profiles = club_profiles[
            (club_profiles['League strength'] >= float(min_strength_cf)) &
            (club_profiles['League strength'] <= float(max_strength_cf))
        ]
        ratio = (club_profiles['League strength'] / target_ls).clip(0.5, 1.2)
        club_profiles['Adjusted Fit %'] = (
            club_profiles['Club Fit %'] * (1 - league_weight_cf) +
            club_profiles['Club Fit %'] * ratio * league_weight_cf
        )
        # penalty if league much stronger
        league_gap = (club_profiles['League strength'] - target_ls).clip(lower=0)
        penalty = (1 - (league_gap / 100)).clip(lower=0.7)
        club_profiles['Adjusted Fit %'] = club_profiles['Adjusted Fit %'] * penalty

        # Market value fit
        value_fit_ratio = (club_profiles['Avg Team Market Value'] / target_market_value).clip(0.5, 1.5)
        value_fit_score = (1 - abs(1 - value_fit_ratio)) * 100

        club_profiles['Final Fit %'] = (
            club_profiles['Adjusted Fit %'] * (1 - market_weight_cf) +
            value_fit_score * market_weight_cf
        )

        results = club_profiles[['Team','League','League strength','Avg Team Market Value',
                                 'Club Fit %','Adjusted Fit %','Final Fit %']].copy()
        results = results.sort_values('Final Fit %', ascending=False).reset_index(drop=True)
        results.insert(0, 'Rank', np.arange(1, len(results)+1))

        st.caption(
            f"Target market value used: €{target_market_value:,.0f} • Native league strength {target_ls:.1f} • Candidate leagues: {len(leagues_cf)}"
        )
        st.dataframe(results.head(int(top_n_cf)), use_container_width=True)

        csv_cf = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download club fit (CSV)", data=csv_cf,
                           file_name=f"club_fit_{player_name.replace(' ','_').lower()}.csv", mime="text/csv")

# ----------------- DOWNLOAD (tables) -----------------
st.subheader("⬇️ Download ranked data")
role_pick = st.selectbox("Role to export", list(ROLES.keys()))
export_view = df_f.sort_values(f"{role_pick} Score", ascending=False)
export_cols = ["Player","Team","League","Age","Contract expires","Market value","League Strength", f"{role_pick} Score"]
csv = export_view[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"scouting_{role_pick.replace(' ','_').lower()}.csv", mime="text/csv")



