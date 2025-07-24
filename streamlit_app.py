import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load data ===
train_df = pd.read_csv("CFB.csv")
predict_df = pd.read_csv("CFB25.csv")
predict_df.rename(columns={"TEAM": "Team"}, inplace=True)

# Load team logos
logos_df = pd.read_csv("team_logos.csv")

# === Train model ===
X_train = train_df[['2023 Power', '2024 RP%', '2024 RR', '2024 TPR', '2024 SOS']]
Y_train = train_df['2024 Power']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, Y_train)

# Calculate residual std for confidence intervals
train_preds = ridge.predict(X_train_scaled)
resid_std = np.std(Y_train - train_preds)

# === Predict ===
X_predict = predict_df[['2023 Power', '2024 RP%', '2024 RR', '2024 TPR', '2024 SOS']]
X_predict_scaled = scaler.transform(X_predict)

predict_df['Predicted 2025 SP+'] = ridge.predict(X_predict_scaled)

# Rescale SP+ scores
SP_PLUS_MIN = -10
SP_PLUS_MAX = 30

min_pred = predict_df['Predicted 2025 SP+'].min()
max_pred = predict_df['Predicted 2025 SP+'].max()

predict_df['Rescaled 2025 SP+'] = (
    (predict_df['Predicted 2025 SP+'] - min_pred) / (max_pred - min_pred)
) * (SP_PLUS_MAX - SP_PLUS_MIN) + SP_PLUS_MIN

predict_df['Rescaled 2025 SP+'] = predict_df['Rescaled 2025 SP+'].round(1)

# Sort for display
ranked_df = predict_df[['Team', 'Rescaled 2025 SP+']].sort_values(by='Rescaled 2025 SP+', ascending=False).reset_index(drop=True)
ranked_df['Rank'] = ranked_df.index + 1
ranked_df = ranked_df[['Rank', 'Team', 'Rescaled 2025 SP+']]

# Merge logos
ranked_df = ranked_df.merge(logos_df, on='Team', how='left')

# === Streamlit UI ===
st.set_page_config(page_title="CFB Spread Predictor", layout="centered")
st.title("🏈 CFB Spread Predictor (2025)")
st.markdown("Select two teams below to calculate a projected point spread and 95% confidence interval.")

team1 = st.selectbox("Team 1", ranked_df["Team"])
team2 = st.selectbox("Team 2", ranked_df["Team"])
home_team = st.radio("Who is the home team?", [team1, team2, "Neutral Site"])

if st.button("Calculate Spread"):
    sp1 = float(ranked_df[ranked_df["Team"] == team1]["Rescaled 2025 SP+"].values[0])
    sp2 = float(ranked_df[ranked_df["Team"] == team2]["Rescaled 2025 SP+"].values[0])
    spread = sp1 - sp2

    if home_team == team1:
        spread += 2.5
    elif home_team == team2:
        spread -= 2.5

    ci_low = round(spread - 1.96 * resid_std, 1)
    ci_high = round(spread + 1.96 * resid_std, 1)

    # Get logos
    logo1 = ranked_df[ranked_df["Team"] == team1]["Logo"].values[0]
    logo2 = ranked_df[ranked_df["Team"] == team2]["Logo"].values[0]

    # Display logos and matchup
    st.subheader("🏆 Projected Spread")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="text-align: center;">
                <img src="{logo1}" width="60"/><br><strong>{team1}</strong>
            </div>
            <div style="font-size: 24px;">vs</div>
            <div style="text-align: center;">
                <img src="{logo2}" width="60"/><br><strong>{team2}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if home_team != "Neutral Site":
        st.markdown(f"🏠 Home Field Advantage: **{home_team}** (+2.5 pts)")

    st.markdown(f"📊 **Spread:** `{team1} {'-' if spread >= 0 else '+'}{abs(round(spread, 1))}`")
    st.markdown(f"🔒 **95% Confidence Interval:** ({ci_low}, {ci_high})")

st.markdown("---")
if st.checkbox("Show full SP+ rankings with logos"):
    display_df = ranked_df.copy()
    display_df['Team'] = display_df.apply(
        lambda row: f'<img src="{row.Logo}" width="40"/> {row.Team}', axis=1
    )
    display_df = display_df[['Rank', 'Team', 'Rescaled 2025 SP+']]
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
