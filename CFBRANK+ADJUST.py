import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load data ===
train_df = pd.read_csv("CFB.csv")
predict_df = pd.read_csv("CFB25.csv")
predict_df.rename(columns={"TEAM": "Team"}, inplace=True)

# Load team logos (expects columns: Team, Logo)
logos_df = pd.read_csv("team_logos.csv")

# === Train model (prior) ===
X_train = train_df[['2023 Power', '2024 RP%', '2024 RR', '2024 TPR', '2024 SOS']]
Y_train = train_df['2024 Power']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, Y_train)

# ========= Weekly update parameters & helper functions (public defaults) =========
HFA_POINTS = 2.5

# Global movement: conservative
K0 = 0.07
BLOWOUT_CLIP = 16
GARBAGE_TIME_MARGIN = 24
GARBAGE_DAMP = 0.60

# Opponent weighting
OPP_STRENGTH_RANGE = 50.0
OPP_WEIGHT_MIN, OPP_WEIGHT_MAX = 0.90, 1.20

# Fixed public defaults (no sliders)
ELITE_GAIN_SCALE = 1.12     # elite-vs-elite boost scale
MAX_GAME_MOVE    = 1.80     # max rating change (power pts) per team per game
PRIOR_W1         = 0.83     # Week 1 prior weight

def expected_margin(r_team, r_opp, venue):
    m = r_team - r_opp
    if venue == "H":
        m += HFA_POINTS
    elif venue == "A":
        m -= HFA_POINTS
    return m

def opponent_weight(r_opp, r_mean):
    w = 1.0 + (r_opp - r_mean) / OPP_STRENGTH_RANGE
    return max(OPP_WEIGHT_MIN, min(OPP_WEIGHT_MAX, w))

def shrink_K(base_K, games_played):
    return base_K / np.sqrt(games_played + 1)

def clip_and_dampen(residual, actual_margin):
    r = np.sign(residual) * min(abs(residual), BLOWOUT_CLIP)
    if abs(actual_margin) > GARBAGE_TIME_MARGIN:
        r *= GARBAGE_DAMP
    return r

# Elite boost only if BOTH teams are elite AND expected spread was very close (<=4)
def elite_multiplier(r_team, r_opp, raw_residual, exp_margin, r_mean, r_std, base_gain=1.05):
    top_cut = r_mean + 1.25 * r_std
    if not (r_team > top_cut and r_opp > top_cut and abs(exp_margin) <= 4):
        return 1.0
    blow = abs(raw_residual)
    if blow >= 28:
        return 1.10 * base_gain
    elif blow >= 21:
        return 1.06 * base_gain
    elif blow >= 14:
        return 1.03 * base_gain
    return 1.0

def prior_weight_for_week(week_through, w1=0.85):
    if not week_through or week_through < 1:
        return float(w1)
    return max(0.30, float(w1) - 0.10 * (week_through - 1))

# Residual std for CI (proxy from training)
train_preds = ridge.predict(X_train_scaled)
resid_std = np.std(Y_train - train_preds)

# === Predict prior (un-updated) power ===
X_predict = predict_df[['2023 Power', '2024 RP%', '2024 RR', '2024 TPR', '2024 SOS']]
X_predict_scaled = scaler.transform(X_predict)
predict_df['Predicted 2025 SP+'] = ridge.predict(X_predict_scaled)  # power space

# === Freeze display scaling to preseason range ===
SP_PLUS_MIN = -10
SP_PLUS_MAX = 30
pre_min = predict_df['Predicted 2025 SP+'].min()
pre_max = predict_df['Predicted 2025 SP+'].max()

# === Streamlit UI ===
st.set_page_config(page_title="CFB Spread Predictor", layout="centered")
st.title("🏈 CFB Spread Predictor (2025)")
st.markdown("Preseason files from GitHub. Weekly results from **Google Sheets** (published CSV).")

# ---- Weekly results from Google Sheets (no upload; minimal UI) ----
st.markdown("### 📅 Weekly Performance Updates (Google Sheets)")
RESULTS_SOURCE = st.secrets.get("RESULTS_CSV_URL", "")
apply_weeks = st.checkbox("Apply these weekly updates to the ratings before spreads/rankings", value=True)
refresh_click = st.button("🔄 Refresh results")

@st.cache_data(ttl=300)
def load_results_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

results_df = None
week_through = None

if RESULTS_SOURCE:
    try:
        if refresh_click:
            load_results_csv.clear()
        results_df = load_results_csv(RESULTS_SOURCE)

        required_cols = {'Week', 'Team', 'Opponent', 'Team_Score', 'Opp_Score', 'Venue'}
        if not required_cols.issubset(results_df.columns):
            st.error(f"Results CSV must include columns: {sorted(list(required_cols))}")
            results_df = None
        else:
            results_df['Week'] = results_df['Week'].astype(int)
            week_through = int(results_df['Week'].max())
            st.success(f"✅ Weekly adjustments are up to date (through Week {week_through}).")

    except Exception as e:
        st.error(f"Could not load weekly results from Google Sheets: {e}")
        results_df = None
else:
    st.warning("Set RESULTS_CSV_URL in .streamlit/secrets.toml to a Google Sheets 'Publish to web' CSV link.")
    results_df = None

# === Build UPDATED ratings in model space ===
ratings = predict_df.set_index('Team')['Predicted 2025 SP+'].to_dict()
games_played = {t: 0 for t in ratings.keys()}

if apply_weeks and (results_df is not None) and (week_through is not None):
    try:
        use_df = results_df[results_df['Week'] <= week_through].copy()
        use_df = use_df.sort_values(['Week', 'Team'])

        global_mean = np.mean(list(ratings.values()))
        global_std  = np.std(list(ratings.values()))

        for _, row in use_df.iterrows():
            team = row['Team']; opp = row['Opponent']
            if team not in ratings or opp not in ratings:
                continue

            venue = str(row['Venue']).strip().upper()
            try:
                team_pts = float(row['Team_Score']); opp_pts = float(row['Opp_Score'])
            except Exception:
                continue

            actual_margin = team_pts - opp_pts
            r_team = ratings[team]; r_opp = ratings[opp]
            exp_margin = expected_margin(r_team, r_opp, venue)

            raw_residual = actual_margin - exp_margin
            residual = clip_and_dampen(raw_residual, actual_margin)
            residual *= elite_multiplier(r_team, r_opp, raw_residual, exp_margin, global_mean, global_std,
                                         base_gain=ELITE_GAIN_SCALE)

            K_team = shrink_K(K0, games_played[team]); K_opp = shrink_K(K0, games_played[opp])
            w_team = opponent_weight(r_opp, global_mean); w_opp = opponent_weight(r_team, global_mean)

            delta_team = float(np.clip(K_team * residual * w_team, -MAX_GAME_MOVE, MAX_GAME_MOVE))
            delta_opp  = float(np.clip(-K_opp * residual * w_opp, -MAX_GAME_MOVE, MAX_GAME_MOVE))

            ratings[team] += delta_team
            ratings[opp]  += delta_opp
            games_played[team] += 1; games_played[opp] += 1

        predict_df['Updated Power Raw'] = predict_df['Team'].map(ratings)
        pw = prior_weight_for_week(week_through, w1=PRIOR_W1)
        predict_df['Updated Power'] = pw * predict_df['Predicted 2025 SP+'] + (1 - pw) * predict_df['Updated Power Raw']

    except Exception as e:
        st.error(f"Error applying weekly updates: {e}")
        predict_df['Updated Power'] = predict_df['Predicted 2025 SP+']
else:
    predict_df['Updated Power'] = predict_df['Predicted 2025 SP+']

# === Rescale to display range using FROZEN preseason min/max ===
predict_df['Rescaled 2025 SP+'] = (
    (predict_df['Updated Power'] - pre_min) / (pre_max - pre_min)
) * (SP_PLUS_MAX - SP_PLUS_MIN) + SP_PLUS_MIN
predict_df['Rescaled 2025 SP+'] = predict_df['Rescaled 2025 SP+'].round(1)

# === Rankings (after) ===
ranked_df = predict_df[['Team', 'Rescaled 2025 SP+']].sort_values(
    by='Rescaled 2025 SP+', ascending=False
).reset_index(drop=True)
ranked_df['Rank'] = ranked_df.index + 1
ranked_df = ranked_df[['Rank', 'Team', 'Rescaled 2025 SP+']]

# Merge logos
ranked_df = ranked_df.merge(logos_df, on='Team', how='left')

# ---- Before/after & deltas ----
predict_df['Rescaled Prior SP+'] = (
    (predict_df['Predicted 2025 SP+'] - pre_min) / (pre_max - pre_min)
) * (SP_PLUS_MAX - SP_PLUS_MIN) + SP_PLUS_MIN

prior_ranks = (predict_df[['Team','Rescaled Prior SP+']]
               .sort_values('Rescaled Prior SP+', ascending=False)
               .reset_index(drop=True))
prior_ranks['Rank_Before'] = prior_ranks.index + 1
prior_ranks = prior_ranks[['Team','Rank_Before']]

post_ranks = ranked_df.rename(columns={'Rank':'Rank_After'})[['Team','Rank_After']]

rank_moves = prior_ranks.merge(post_ranks, on='Team', how='inner')
rank_moves['ΔRank'] = rank_moves['Rank_After'] - rank_moves['Rank_Before']     # + = fell in rank (worse)
rank_moves = rank_moves.merge(
    predict_df[['Team','Rescaled Prior SP+','Rescaled 2025 SP+']],
    on='Team', how='left'
)
rank_moves['ΔSP+'] = (rank_moves['Rescaled 2025 SP+'] - rank_moves['Rescaled Prior SP+']).round(1)
rank_moves['Rank_Change'] = rank_moves['Rank_Before'] - rank_moves['Rank_After']  # + = improved

# =========================
#   SPREAD CALCULATOR FIRST
# =========================
st.markdown("---")
st.markdown("### 📐 Project a Spread")

teams_list = list(ranked_df["Team"])
team1 = st.selectbox("Team 1", teams_list, index=0 if len(teams_list) else 0)
team2 = st.selectbox("Team 2", teams_list, index=1 if len(teams_list) > 1 else 0)
home_team = st.radio("Who is the home team?", [team1, team2, "Neutral Site"])

if st.button("Calculate Spread"):
    sp1 = float(ranked_df.loc[ranked_df["Team"] == team1, "Rescaled 2025 SP+"].values[0])
    sp2 = float(ranked_df.loc[ranked_df["Team"] == team2, "Rescaled 2025 SP+"].values[0])
    spread = sp1 - sp2

    if home_team == team1:
        spread += HFA_POINTS
    elif home_team == team2:
        spread -= HFA_POINTS

    ci_low = round(spread - 1.96 * resid_std, 1)
    ci_high = round(spread + 1.96 * resid_std, 1)

    logo1 = ranked_df.loc[ranked_df["Team"] == team1, "Logo"].values[0]
    logo2 = ranked_df.loc[ranked_df["Team"] == team2, "Logo"].values[0]

    st.subheader("🏆 Projected Spread")
    st.markdown(
        f"""
<div style="display:flex; align-items:center; gap:20px;">
  <div style="text-align:center;"><img src="{logo1}" width="60"/><br><strong>{team1}</strong></div>
  <div style="font-size:24px;">vs</div>
  <div style="text-align:center;"><img src="{logo2}" width="60"/><br><strong>{team2}</strong></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if home_team != "Neutral Site":
        st.markdown(f"🏠 Home Field Advantage: **{home_team}** (+{HFA_POINTS} pts)")
    st.markdown(f"📊 **Spread:** `{team1} {'-' if spread >= 0 else '+'}{abs(round(spread, 1))}`")
    st.markdown(f"🔒 **95% Confidence Interval:** ({ci_low}, {ci_high})")

# ---------- Helper to render aligned, color-coded mini tables (no leading spaces) ----------
def render_rank_delta_table(df):
    """Render a compact, aligned, color-coded HTML table for Gainers/Losers.
       Expects columns: Team, Rank, Chg, ΔSP+."""
    def color(v):
        if pd.isna(v) or v == 0:
            return "#6b7280"  # gray
        return "#16a34a" if v > 0 else "#dc2626"  # green / red

    rows = []
    for _, r in df.iterrows():
        chg = r["Chg"]; dsp = r["ΔSP+"]
        chg_txt = f"{'+' if pd.notna(chg) and chg>0 else ''}{int(chg) if pd.notna(chg) else '—'}"
        dsp_txt = f"{dsp:+.1f}" if pd.notna(dsp) else "—"
        rows.append(
            f"<tr>"
            f"<td style='padding:8px 10px; border-bottom:1px solid rgba(0,0,0,0.06);'>{r['Team']}</td>"
            f"<td style='padding:8px 10px; border-bottom:1px solid rgba(0,0,0,0.06); text-align:right;'>{int(r['Rank'])}</td>"
            f"<td style='padding:8px 10px; border-bottom:1px solid rgba(0,0,0,0.06); text-align:right; color:{color(chg)}; font-weight:600;'>{chg_txt}</td>"
            f"<td style='padding:8px 10px; border-bottom:1px solid rgba(0,0,0,0.06); text-align:right; color:{color(dsp)}; font-weight:600;'>{dsp_txt}</td>"
            f"</tr>"
        )
    html = (
        "<div style='width:100%;'>"
        "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
        "<thead><tr>"
        "<th style='text-align:left; padding:8px 10px;'>Team</th>"
        "<th style='text-align:right; padding:8px 10px;'>Rank</th>"
        "<th style='text-align:right; padding:8px 10px;'>Chg</th>"
        "<th style='text-align:right; padding:8px 10px;'>ΔSP+</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ---- Top 5 gainers/losers (aligned, color-coded) ----
if apply_weeks and (results_df is not None) and (week_through is not None):
    gainers = (rank_moves.sort_values(['Rank_Change','ΔSP+'], ascending=False)
               .head(5)[['Team','Rank_After','Rank_Change','ΔSP+']])
    losers  = (rank_moves.sort_values(['Rank_Change','ΔSP+'], ascending=[True, True])
               .head(5)[['Team','Rank_After','Rank_Change','ΔSP+']])
    gainers = gainers.rename(columns={'Rank_After':'Rank','Rank_Change':'Chg'})
    losers  = losers.rename(columns={'Rank_After':'Rank','Rank_Change':'Chg'})

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Gainers (Since Last Update)")
        render_rank_delta_table(gainers)
    with c2:
        st.subheader("Losers (Since Last Update)")
        render_rank_delta_table(losers)

# ---- Full rankings with color-coded change badges ----
def change_badge_html(delta):
    if pd.isna(delta) or int(delta) == 0:
        return "<span style='color:#6b7280;'>—</span>"
    d = int(delta)
    if d < 0:   # improved (moved up)
        return f"<span style='color:#16a34a; font-weight:600;'>▲{abs(d)}</span>"
    else:       # declined (moved down)
        return f"<span style='color:#dc2626; font-weight:600;'>▼{d}</span>"

st.markdown("---")
if st.checkbox("Show full SP+ rankings with logos & change"):
    merged = ranked_df.merge(rank_moves[['Team','ΔRank']], on='Team', how='left')
    display_df = merged.copy()
    display_df['Team'] = display_df.apply(
        lambda row: f'<img src="{row.Logo}" width="40"/> {row.Team}', axis=1
    )
    display_df['Change'] = merged['ΔRank'].apply(change_badge_html)
    display_df = display_df[['Rank', 'Team', 'Change', 'Rescaled 2025 SP+']]
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
