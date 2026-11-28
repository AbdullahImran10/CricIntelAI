
import streamlit as st
import pandas as pd

from src.prediction_service import df, predict_match

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="CricIntelAI",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 CricIntelAI")
st.subheader("AI-Powered T20 Cricket Match Predictor")

st.markdown(
    "Predict match winners using historical cricket performance "
    "and toss information."
)

st.divider()

# =====================================
# INPUT SECTION
# =====================================

teams = sorted(set(df["team1"]) | set(df["team2"]))
venues = sorted(df["venue"].dropna().unique())

latest_date = df["date"].max().date()
default_date = max(
    pd.Timestamp.today().date(),
    latest_date + pd.Timedelta(days=1)
)

st.header("Match Details")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    available_opponents = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2", available_opponents)

col3, col4 = st.columns(2)

with col3:
    venue = st.selectbox("Match Venue", venues)

with col4:
    prediction_date = st.date_input(
        "Prediction Date",
        value=default_date,
        min_value=latest_date + pd.Timedelta(days=1)
    )

col5, col6 = st.columns(2)

with col5:
    toss_winner = st.selectbox(
        "Toss Winner",
        [team1, team2]
    )

with col6:
    toss_decision = st.selectbox(
        "Toss Decision",
        ["bat", "field"]
    )

st.divider()

# =====================================
# PREDICTION
# =====================================

if st.button("🏏 Predict Match Winner", use_container_width=True):

    try:
        with st.spinner("Analyzing historical cricket data..."):

            result = predict_match(
                team1,
                team2,
                venue,
                prediction_date,
                toss_winner,
                toss_decision
            )

        st.success("Prediction completed!")

        st.header("Prediction Results")

        st.subheader(
            f"🏆 Predicted Winner: {result['winner']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                f"{team1} Win Probability",
                f"{result['team1_probability']:.2f}%"
            )

        with col2:
            st.metric(
                f"{team2} Win Probability",
                f"{result['team2_probability']:.2f}%"
            )

        st.subheader("Win Probability Comparison")

        chart_data = pd.DataFrame({
            "Team": [team1, team2],
            "Win Probability (%)": [
                result["team1_probability"],
                result["team2_probability"]
            ]
        }).set_index("Team")

        st.bar_chart(chart_data)

        st.info(
            f"Historical matches analyzed: "
            f"{result['historical_matches']}"
        )

        st.caption(
            "Probabilities are estimates from a machine learning "
            "model, not guarantees of match outcomes."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption("CricIntelAI | Men's T20 International Analytics")
