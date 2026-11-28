
import pandas as pd
import joblib

DATA_PATH = "data/processed/t20_matches_prepared.csv"
MODEL_PATH = "models/logistic_regression.pkl"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

model = joblib.load(MODEL_PATH)


def team_statistics(history, team, opponent, venue):

    matches = history[
        (history["team1"] == team) |
        (history["team2"] == team)
    ].sort_values(["date", "match_id"])

    recent = matches.tail(5)

    recent_form = (
        (recent["winner"] == team).mean()
        if len(recent) else 0
    )

    overall = (
        (matches["winner"] == team).mean()
        if len(matches) else 0
    )

    h2h = history[
        (
            (history["team1"] == team) &
            (history["team2"] == opponent)
        ) |
        (
            (history["team1"] == opponent) &
            (history["team2"] == team)
        )
    ]

    h2h_rate = (
        (h2h["winner"] == team).mean()
        if len(h2h) else 0
    )

    venue_matches = matches[matches["venue"] == venue]

    venue_rate = (
        (venue_matches["winner"] == team).mean()
        if len(venue_matches) else 0
    )

    toss_matches = matches[matches["toss_winner"] == team]

    toss_rate = (
        (toss_matches["winner"] == team).mean()
        if len(toss_matches) else 0
    )

    return {
        "recent_form": recent_form,
        "win_percentage": overall,
        "h2h_percentage": h2h_rate,
        "venue_win_percentage": venue_rate,
        "toss_win_percentage": toss_rate
    }


def predict_match(team1, team2, venue, prediction_date,
                  toss_winner, toss_decision):

    history = df[df["date"] < pd.Timestamp(prediction_date)]

    if history.empty:
        raise ValueError("No historical matches available.")

    stats1 = team_statistics(history, team1, team2, venue)
    stats2 = team_statistics(history, team2, team1, venue)

    toss_advantage = 1 if toss_winner == team1 else -1
    decision_value = 1 if toss_decision == "bat" else -1

    features = pd.DataFrame([{
        "recent_form_difference":
            stats1["recent_form"] - stats2["recent_form"],

        "win_percentage_difference":
            stats1["win_percentage"] - stats2["win_percentage"],

        "h2h_difference":
            stats1["h2h_percentage"] - stats2["h2h_percentage"],

        "venue_difference":
            stats1["venue_win_percentage"] - stats2["venue_win_percentage"],

        "toss_performance_difference":
            stats1["toss_win_percentage"] - stats2["toss_win_percentage"],

        "toss_advantage": toss_advantage,

        "toss_decision_advantage":
            toss_advantage * decision_value
    }])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    probability_map = dict(zip(model.classes_, probabilities))

    team1_probability = probability_map[1] * 100
    team2_probability = probability_map[0] * 100

    winner = team1 if prediction == 1 else team2

    return {
        "winner": winner,
        "team1_probability": team1_probability,
        "team2_probability": team2_probability,
        "historical_matches": len(history)
    }
