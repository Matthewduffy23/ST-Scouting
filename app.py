# app.py — Advanced Scouting Tool (presets, colored role table, attacker polar chart,
# adjustable comparison pool, improved NPG90 vs xG90 scatter, league-weighted player role scores
# toggle for single player, styled strengths/weaknesses + style derived from extra metrics)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))

    # Global age default 16–40
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 40))

    pos_text = st.text_input("Position startswith", "CF")

    # Defaults OFF
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.4, 0.05)  # default 0.4

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
    value_band_max = st.number_input("Value band (tab 4 max €)", min_value=0, value=min_value if min_value>0 else 5_000_000, step=250_000)

    st.subheader("Minimum performance thresholds")
    enable_min_perf = st.checkbox("Require minimum percentile on selected metrics", value=False)
    sel_metrics = st.multiselect("Metrics to threshold", FEATURES[:], default=['Non-penalty goals per 90','xG per 90'] if enable_min_perf else [])
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

# ----------------- FILTER POOL -----------------
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
    age_min_pool, age_max_pool = c3.slider("Pool age", 14, 45, (16, 40))  # default max 40
    same_pos = st.checkbox("Limit pool to current position prefix", value=True)
    # Player-specific league weighting toggle (OFF by default). Beta default 0.4
    c4, c5 = st.columns([1.2, 2])
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

def percentile_in_series(value: float, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    rank = (s < float(value)).mean() * 100.0
    eq_share = (s == float(value)).mean() * 100.0
    return min(100.0, rank + 0.5 * eq_share)

def percentiles_for_player_in_pool(pool_df: pd.DataFrame, ply_row: pd.Series, metrics: list) -> dict:
    if pool_df.empty:
        return {}
    out = {}
    for m in metrics:
        if m in pool_df.columns:
            out[m] = percentile_in_series(ply_row[m], pool_df[m])
    return {k:v for k,v in out.items() if pd.notna(v)}

def player_role_scores_from_pct(pct_map: dict, *, player_league_strength: float, use_weight: bool, beta: float) -> dict:
    out = {}
    for role, rd in ROLES.items():
        weights = rd["metrics"]
        total = sum(weights.values()) or 1.0
        metric_score = 0.0
        for m, w in weights.items():
            if m in pct_map:
                metric_score += pct_map[m] * w
        metric_score = metric_score / total
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

if player_row.empty:
    st.info("Pick a player above.")
else:
    meta = player_row[["Team","League","Age","Contract expires","League Strength","Market value"]].iloc[0]
    st.caption(
        f"**{player_name}** — {meta['Team']} • {meta['League']} • Age {int(meta['Age'])} • "
        f"Contract: {pd.to_datetime(meta['Contract expires']).date() if pd.notna(meta['Contract expires']) else 'N/A'} • "
        f"League Strength {meta['League Strength']:.1f} • Value €{meta['Market value']:,.0f}"
    )

    # Build pool & compute player percentiles within that pool (for polar metrics)
    pool_df = build_pool_df()
    if pool_df.empty:
        st.warning("Comparison pool is empty. Add at least one league.")
    else:
        ply = player_row.iloc[0]
        pct_map = percentiles_for_player_in_pool(pool_df, ply, POLAR_METRICS)

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

# ----------------- SCATTER: NPG/90 (X) vs xG/90 (Y) on the same pool -----------------
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

    # all points — black, slightly larger circles (no minutes sizing)
    ax2.scatter(scat[x_col], scat[y_col], s=42, c="black", alpha=0.8,
                edgecolors="white", linewidths=0.5)

    # highlight selected player
    if player_name in scat["Player"].values:
        pr = scat[scat["Player"] == player_name].iloc[0]
        ax2.scatter([pr[x_col]], [pr[y_col]], s=170, c="red", edgecolors="white", linewidths=1.2, zorder=5)
        ax2.annotate(player_name, (pr[x_col], pr[y_col]), xytext=(10,10), textcoords="offset points",
                     color="red", fontsize=10, weight="bold")

    # median lines
    ax2.axvline(x_med, color="#9ca3af", linestyle="--", linewidth=1.2, zorder=0)
    ax2.axhline(y_med, color="#9ca3af", linestyle="--", linewidth=1.2, zorder=0)

    # limits & labels (bold)
    ax2.set_xlim(0, max(1.0, float(scat[x_col].max()) * 1.05))
    ax2.set_ylim(0, max(1.0, float(scat[y_col].max()) * 1.05))
    ax2.set_xlabel("Non-penalty goals per 90", fontweight="bold")
    ax2.set_ylabel("xG per 90", fontweight="bold")
    ax2.set_title("Shot output vs shot quality (medians shown; grey background)", fontweight="bold")

    st.pyplot(fig2, use_container_width=True)
else:
    st.info("Add at least one league to the comparison pool (and ensure required metrics exist).")

# ----------------- NOTES (Style + strengths/weaknesses from extra metrics) -----------------
st.subheader("📝 Notes")

EXTRA_METRICS = [
    'Defensive duels per 90','Aerial duels per 90','Aerial duels won, %',
    'Non-penalty goals per 90','xG per 90','Shots per 90','Goal conversion, %',
    'Crosses per 90','Accurate crosses, %','Dribbles per 90','Successful dribbles, %',
    'Touches in box per 90','Progressive runs per 90','Passes per 90','Accurate passes, %',
    'xA per 90','Passes to penalty area per 90','Deep completions per 90','Smart passes per 90'
]

STYLE_MAP = {
    'Defensive duels per 90': {'style':'High work rate','sw':'Defensive Duels'},
    'Aerial duels won, %': {'style':'Target Man','sw':'Aerial Duels'},
    'Aerial duels per 90': {'style':'Focal point','sw':'Aerial volume'},
    'Non-penalty goals per 90': {'style':None,'sw':'Scoring Goals'},
    'xG per 90': {'style':'Gets into good goal scoring positions','sw':'Attacking Positioning'},
    'Shots per 90': {'style':'Takes many shots','sw':'Shot Volume'},
    'Goal conversion, %': {'style':None,'sw':'Finishing'},
    'Crosses per 90': {'style':'Crosser','sw':'Crossing Volume'},
    'Accurate crosses, %': {'style':None,'sw':'Crossing Accuracy'},
    'Dribbles per 90': {'style':'1v1 dribbler','sw':'Dribble Volume'},
    'Successful dribbles, %': {'style':None,'sw':'Dribbling Efficiency'},
    'Touches in box per 90': {'style':'Busy in the box','sw':'Penalty-box Coverage'},
    'Progressive runs per 90': {'style':'Ball carrier','sw':'Progressive Runs'},
    'Passes per 90': {'style':'Build-up involvement','sw':'Build-up Volume'},
    'Accurate passes, %': {'style':None,'sw':'Retention'},
    'xA per 90': {'style':'Chance creator','sw':'Chance Creation'},
    'Passes to penalty area per 90': {'style':None,'sw':'Passes to Penalty Area'},
    'Deep completions per 90': {'style':None,'sw':'Deep Completions'},
    'Smart passes per 90': {'style':None,'sw':'Smart Passes'},
}

HI = 75
LO = 25
STYLE_T = 65

def chips(items, color):
    if not items: return "_None identified._"
    spans = [f"<span style='background:{color};color:#111;padding:2px 6px;border-radius:10px;margin:0 6px 6px 0;display:inline-block'>{txt}</span>"
             for txt in items[:10]]
    return " ".join(spans)

if not player_row.empty:
    ply = player_row.iloc[0]

    # Percentiles for EXTRA_METRICS: prefer pool-based; fallback to league-based percentiles
    pct_extra = {}
    if 'pool_df' in locals() and not pool_df.empty:
        for m in EXTRA_METRICS:
            if m in df.columns:
                pct_extra[m] = percentile_in_series(ply[m], pool_df[m])
    for m in EXTRA_METRICS:
        if m not in pct_extra or pd.isna(pct_extra[m]):
            col = f"{m} Percentile"
            if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                pct_extra[m] = float(player_row[col].iloc[0])

    strengths, weaknesses, styles = [], [], []
    for m, v in pct_extra.items():
        lab = STYLE_MAP.get(m, {})
        sw_name = lab.get('sw', m)
        style_tag = lab.get('style', None)

        if v >= HI: strengths.append((sw_name, v))
        elif v <= LO: weaknesses.append((sw_name, v))
        if style_tag and v >= STYLE_T: styles.append((style_tag, v))

    # Sort by percentile but show only labels (no 95th etc.)
    strengths = [name for name,_ in sorted(strengths, key=lambda x: -x[1])]
    weaknesses = [name for name,_ in sorted(weaknesses, key=lambda x: x[1])]
    styles = [name for name,_ in sorted({(n,v) for n,v in styles}, key=lambda x: -x[1])]

    st.markdown(
        f"**Profile:** {player_name} — {ply['Team']} ({ply['League']}), "
        f"age {int(ply['Age'])}, minutes {int(ply['Minutes played'])}."
    )

    # Best role from the FIRST THREE roles only
    first_three = list(ROLES.keys())[:3]
    best_line = ""
    if 'role_scores' in locals() and role_scores:
        subset = {k: v for k, v in role_scores.items() if k in first_three}
        if subset:
            best_role = max(subset.items(), key=lambda kv: kv[1])[0]
            best_line = f"**Best role:** {best_role}."
    else:
        # fallback to table-based scores
        best_role, best_val = None, -1
        for r in first_three:
            col = f"{r} Score"
            if col in player_row.columns and pd.notna(player_row[col].iloc[0]):
                if player_row[col].iloc[0] > best_val:
                    best_val = float(player_row[col].iloc[0]); best_role = r
        if best_role is not None:
            best_line = f"**Best role (league):** {best_role}."
    if best_line:
        st.markdown(best_line)

    st.markdown("**Style:**")
    st.markdown(chips(styles, "#bfdbfe"), unsafe_allow_html=True)  # light blue

    st.markdown("**Strengths:**")
    st.markdown(chips(strengths, "#a7f3d0"), unsafe_allow_html=True)  # light green

    st.markdown("**Weaknesses / growth areas:**")
    st.markdown(chips(weaknesses, "#fecaca"), unsafe_allow_html=True)  # light red
else:
    st.caption("Pick a player above to generate notes.")

# ----------------- DOWNLOAD -----------------
st.subheader("⬇️ Download ranked data")
role_pick = st.selectbox("Role to export", list(ROLES.keys()))
export_view = df_f.sort_values(f"{role_pick} Score", ascending=False)
export_cols = ["Player","Team","League","Age","Contract expires","Market value","League Strength", f"{role_pick} Score"]
csv = export_view[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"scouting_{role_pick.replace(' ','_').lower()}.csv", mime="text/csv")


