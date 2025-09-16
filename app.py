# advanced_scouting_app.py — Advanced Scouting Tool (with role descriptions + single-player percentile chart)
# Filters: age, minutes, league quality, contract, market value, minimum performance thresholds
# League-weighted role scoring (toggle + adjustable beta)
# Four output tables in tabs + single-player role profile (Attacking / Defensive / Possession bar chart)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Scouting Tool", layout="wide")
st.title("🔎 Advanced Scouting Tool")

# ----------------- CONFIG -----------------
INCLUDED_LEAGUES = [
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

# Categories for the single-player bar chart (use any metrics present in your data)
METRIC_GROUPS = {
    "Attacking": [
        'Crosses per 90', 'Accurate crosses, %',
        'Non-penalty goals per 90', 'xG per 90', 'Goal conversion, %',
        'Head goals per 90', 'Key passes per 90', 'Shots per 90',
        'Shots on target, %', 'Touches in box per 90',
    ],
    "Defensive": [
        'Aerial duels per 90', 'Aerial duels won, %',
        'Defensive duels per 90', 'Defensive duels won, %',
        'PAdj Interceptions'
    ],
    "Possession": [
        'Accelerations per 90', 'Dribbles per 90', 'Successful dribbles, %',
        'Key passes per 90', 'Passes per 90', 'Accurate passes, %',
        'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
        'Smart passes per 90', 'Progressive runs per 90'
    ],
}

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
        'desc': "Wall passes, entries to PA, combine & create under pressure.",
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

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    leagues_sel = st.multiselect("Leagues", leagues_avail, default=INCLUDED_LEAGUES)

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 5000, (500, 5000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 33))

    pos_text = st.text_input("Position startswith", "CF")

    apply_contract = st.checkbox("Filter by contract expiry", value=True)
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=True)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.7, 0.05, help="0 = ignore league strength; 1 = only league strength")

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

# ----------------- PERCENTILES (per league) -----------------
for c in FEATURES:
    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
df_f = df_f.dropna(subset=FEATURES)

for feat in FEATURES:
    df_f[f"{feat} Percentile"] = df_f.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ----------------- ROLE SCORING -----------------
def compute_weighted_role_score(df_in: pd.DataFrame, role_name: str, metrics: dict, beta: float, league_weighting: bool) -> pd.Series:
    total_w = sum(metrics.values()) if metrics else 1.0
    wsum = np.zeros(len(df_in))
    for m, w in metrics.items():
        col = f"{m} Percentile"
        if col in df_in.columns:
            wsum += df_in[col].values * w
    player_score = wsum / total_w  # 0..100

    if league_weighting:
        league_scaled = (df_in["League Strength"].fillna(50) / 100.0) * 100.0
        final = (1 - beta) * player_score + beta * league_scaled
    else:
        final = player_score

    return final

for role_name, role_def in ROLES.items():
    df_f[f"{role_name} Score"] = compute_weighted_role_score(
        df_f, role_name, role_def["metrics"], beta=beta, league_weighting=use_league_weighting
    )

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

# ----------------- OUTPUT TABLES -----------------
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

tabs = st.tabs(["Overall Top N", "U23 Top N", "Expiring Contracts", "Value Band (≤ max €)"])

for role, role_def in ROLES.items():
    st.markdown(f"### Role: **{role}** {'(league-weighted)' if use_league_weighting else '(unweighted)'}")
    st.caption(role_def.get("desc", ""))

    t1, t2, t3, t4 = tabs

    with t1:
        st.markdown(f"**Overall Top {int(top_n)}**")
        st.dataframe(top_table(df_f, role, int(top_n)), use_container_width=True)

    with t2:
        u23_cutoff = st.number_input(f"U23 cutoff for {role}", min_value=16, max_value=30, value=23, step=1, key=f"u23_{role}")
        view = filtered_view(df_f, age_max=u23_cutoff)
        st.dataframe(top_table(view, role, int(top_n)), use_container_width=True)

    with t3:
        exp_year = st.number_input(f"Expiring by year for {role}", min_value=2024, max_value=2030, value=cutoff_year, step=1, key=f"exp_{role}")
        view = filtered_view(df_f, contract_year=exp_year)
        st.dataframe(top_table(view, role, int(top_n)), use_container_width=True)

    with t4:
        v_max = st.number_input(f"Max value (€) for {role}", min_value=0, value=value_band_max, step=100_000, key=f"val_{role}")
        view = filtered_view(df_f, value_max=v_max)
        st.dataframe(top_table(view, role, int(top_n)), use_container_width=True)

    st.divider()

# ----------------- SINGLE PLAYER ROLE PROFILE + LEAGUE PERCENTILE CHART -----------------
st.subheader("🎯 Single Player Role Profile")
player_name = st.selectbox("Choose player", sorted(df_f["Player"].unique()))
player_row = df_f[df_f["Player"] == player_name].head(1)

def _bar_color(v):
    # simple traffic-light color by percentile
    if v >= 66:   return "#22c55e"  # green
    if v >= 33:   return "#eab308"  # yellow
    return "#f97316"                # orange

def plot_player_percentiles(player: pd.DataFrame):
    # Use precomputed league percentiles in df_f
    # Build three panels (Attacking / Defensive / Possession)
    fig = plt.figure(figsize=(12, 8), dpi=180)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.9, 1.1], hspace=0.35)

    groups = list(METRIC_GROUPS.keys())
    for gi, group in enumerate(groups):
        ax = fig.add_subplot(gs[gi, 0])
        metrics = [m for m in METRIC_GROUPS[group] if f"{m} Percentile" in df_f.columns]
        if not metrics:
            ax.axis("off")
            continue

        vals = [float(player[f"{m} Percentile"].iloc[0]) for m in metrics]
        y = np.arange(len(metrics))

        # background grid bands every 10
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(metrics)-0.5)
        for x in range(0, 110, 10):
            ax.axvline(x, color="#e5e7eb", lw=0.8, zorder=0)

        # 50th percentile dashed
        ax.axvline(50, color="#111827", lw=1.2, ls="--", alpha=0.7, zorder=1)

        # bars
        colors = [_bar_color(v) for v in vals]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="#111827", linewidth=0.4)

        # labels
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=9)
        ax.set_xlabel("Percentile Rank", fontsize=9)
        ax.set_title(group, loc="left", fontsize=13, fontweight="bold")

    fig.tight_layout()
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

    # small table of role percentiles
    rows = []
    for role in ROLES.keys():
        col = f"{role} Score"
        score = player_row[col].iloc[0] if col in player_row.columns else np.nan
        rows.append({"Role": role, "Percentile": int(round(score)) if pd.notna(score) else None})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # percentile bar chart
    fig = plot_player_percentiles(player_row)
    st.pyplot(fig, use_container_width=True)

# ----------------- DOWNLOAD -----------------
st.subheader("⬇️ Download ranked data")
role_pick = st.selectbox("Role to export", list(ROLES.keys()))
export_view = df_f.sort_values(f"{role_pick} Score", ascending=False)
export_cols = ["Player","Team","League","Age","Contract expires","Market value","League Strength", f"{role_pick} Score"]
csv = export_view[export_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"scouting_{role_pick.replace(' ','_').lower()}.csv", mime="text/csv")






