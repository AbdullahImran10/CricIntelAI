
import pandas as pd
import joblib
from datetime import date

# ==========================================
# 1. LOAD DATA AND MODEL
# ==========================================

DATA_PATH = "data/processed/t20_matches_prepared.csv"
MODEL_PATH = "models/logistic_regression.pkl"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

model = joblib.load(MODEL_PATH)

print("\n========== CRICINTELAI PREDICTION ==========")
print("Historical matches:", len(df))


# ==========================================
# 2. HISTORICAL STATISTICS
# ==========================================

def team_statistics(history, team, opponent, venue):

    team_matches = history[
        (history["team1"] == team) |
        (history["team2"] == team)
    ].sort_values(["date", "match_id"])

    # Recent form: last 5 matches
    recent = team_matches.tail(5)

    recent_form = (
        (recent["winner"] == team).mean()
        if len(recent) > 0 else 0
    )

    # Overall win percentage
    overall_win = (
        (team_matches["winner"] == team).mean()
        if len(team_matches) > 0 else 0
    )

    # Head-to-head performance
    h2h = history[
        (
            ((history["team1"] == team) &
             (history["team2"] == opponent)) |
            ((history["team1"] == opponent) &
             (history["team2"] == team))
        )
    ]

    h2h_win = (
        (h2h["winner"] == team).mean()
        if len(h2h) > 0 else 0
    )

    # Venue performance
    venue_matches = team_matches[
        team_matches["venue"] == venue
    ]

    venue_win = (
        (venue_matches["winner"] == team).mean()
        if len(venue_matches) > 0 else 0
    )

    # Performance when winning the toss
    toss_matches = team_matches[
        team_matches["toss_winner"] == team
    ]

    toss_win = (
        (toss_matches["winner"] == team).mean()
        if len(toss_matches) > 0 else 0
    )

    return {
        "recent_form": recent_form,
        "win_percentage": overall_win,
        "h2h_percentage": h2h_win,
        "venue_win_percentage": venue_win,
        "toss_win_percentage": toss_win
    }


# ==========================================
# 3. USER INPUT
# ==========================================

teams = sorted(set(df["team1"]) | set(df["team2"]))
venues = sorted(df["venue"].dropna().unique())

print("\nAvailable teams:")
print(", ".join(teams))

team1 = input("\nEnter Team 1: ").strip()
team2 = input("Enter Team 2: ").strip()
venue = input("Enter venue: ").strip()

prediction_date_input = input(
    f"Prediction date (YYYY-MM-DD, default {date.today()}): "
).strip()

prediction_date = (
    pd.to_datetime(prediction_date_input)
    if prediction_date_input
    else pd.Timestamp(date.today())
)

toss_winner = input("Toss winner: ").strip()

toss_decision = input("Toss decision (bat/field): ").strip().lower()


# ==========================================
# 4. VALIDATE INPUT
# ==========================================

def find_match(value, options):
    return next(
        (item for item in options if item.lower() == value.lower()),
        None
    )


team1 = find_match(team1, teams)
team2 = find_match(team2, teams)
venue = find_match(venue, venues)

if not team1 or not team2 or team1 == team2:
    raise ValueError("Please enter two different valid teams.")

if not venue:
    raise ValueError("Invalid venue.")

if toss_winner.lower() not in [team1.lower(), team2.lower()]:
    raise ValueError("Toss winner must be one of the two teams.")

toss_winner = team1 if toss_winner.lower() == team1.lower() else team2

if toss_decision not in ["bat", "field"]:
    raise ValueError("Toss decision must be bat or field.")

if prediction_date <= df["date"].max():
    print("Warning: Prediction date is not after the latest dataset match.")


# ==========================================
# 5. CALCULATE HISTORICAL FEATURES
# ==========================================

history = df[df["date"] < prediction_date].copy()

if history.empty:
    raise ValueError("No historical matches available for this date.")

stats1 = team_statistics(history, team1, team2, venue)
stats2 = team_statistics(history, team2, team1, venue)

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

    "toss_advantage":
        1 if toss_winner == team1 else -1,

    "toss_decision_advantage": (
        (1 if toss_winner == team1 else -1)
        * (1 if toss_decision == "bat" else -1)
    )
}])


# ==========================================
# 6. PREDICT MATCH WINNER
# ==========================================

prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

probability_map = dict(zip(model.classes_, probabilities))

team1_probability = probability_map[1] * 100
team2_probability = probability_map[0] * 100

predicted_winner = team1 if prediction == 1 else team2


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== MATCH PREDICTION ==========")

print(f"Match: {team1} vs {team2}")
print(f"Venue: {venue}")
print(f"Toss winner: {toss_winner}")
print(f"Toss decision: {toss_decision}")
print(f"Historical matches considered: {len(history)}")

print("\n---------- WIN PROBABILITIES ----------")

print(f"{team1}: {team1_probability:.2f}%")
print(f"{team2}: {team2_probability:.2f}%")

print("\n---------- FINAL PREDICTION ----------")

print("Predicted Winner:", predicted_winner)

print("\n======================================")
