
import pandas as pd

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

df = pd.read_csv("data/processed/t20_matches_prepared.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)


# ==========================================
# 2. INITIALIZE HISTORICAL RECORDS
# ==========================================

team_history = {}
head_to_head = {}
venue_history = {}

# Stores match results when a team previously
# won the toss
toss_history = {}


# ==========================================
# 3. INITIALIZE FEATURE LISTS
# ==========================================

team1_form = [0.0] * len(df)
team2_form = [0.0] * len(df)

team1_win_percentage = [0.0] * len(df)
team2_win_percentage = [0.0] * len(df)

team1_h2h = [0.0] * len(df)
team2_h2h = [0.0] * len(df)

team1_venue_percentage = [0.0] * len(df)
team2_venue_percentage = [0.0] * len(df)

team1_toss_percentage = [0.0] * len(df)
team2_toss_percentage = [0.0] * len(df)


# ==========================================
# 4. PROCESS MATCHES CHRONOLOGICALLY
# ==========================================

for date, daily_matches in df.groupby("date", sort=True):

    # --------------------------------------
    # FIRST: CALCULATE HISTORICAL FEATURES
    # --------------------------------------

    for index, match in daily_matches.iterrows():

        team1 = match["team1"]
        team2 = match["team2"]
        venue = match["venue"]

        history1 = team_history.get(team1, [])
        history2 = team_history.get(team2, [])

        # FEATURE 1: RECENT FORM (LAST 5)

        last5_team1 = history1[-5:]
        last5_team2 = history2[-5:]

        team1_form[index] = (
            sum(last5_team1) / len(last5_team1)
            if last5_team1 else 0
        )

        team2_form[index] = (
            sum(last5_team2) / len(last5_team2)
            if last5_team2 else 0
        )

        # FEATURE 2: OVERALL WIN PERCENTAGE

        team1_win_percentage[index] = (
            sum(history1) / len(history1)
            if history1 else 0
        )

        team2_win_percentage[index] = (
            sum(history2) / len(history2)
            if history2 else 0
        )

        # FEATURE 3: HEAD-TO-HEAD

        pair = tuple(sorted([team1, team2]))

        h2h_history = head_to_head.get(pair, [])

        team1_h2h[index] = (
            sum(1 for result in h2h_history if result == team1)
            / len(h2h_history)
            if h2h_history else 0
        )

        team2_h2h[index] = (
            sum(1 for result in h2h_history if result == team2)
            / len(h2h_history)
            if h2h_history else 0
        )

        # FEATURE 4: VENUE PERFORMANCE

        venue_team1 = venue_history.get((team1, venue), [])
        venue_team2 = venue_history.get((team2, venue), [])

        team1_venue_percentage[index] = (
            sum(venue_team1) / len(venue_team1)
            if venue_team1 else 0
        )

        team2_venue_percentage[index] = (
            sum(venue_team2) / len(venue_team2)
            if venue_team2 else 0
        )

        # FEATURE 5: HISTORICAL TOSS PERFORMANCE

        toss_team1 = toss_history.get(team1, [])
        toss_team2 = toss_history.get(team2, [])

        team1_toss_percentage[index] = (
            sum(toss_team1) / len(toss_team1)
            if toss_team1 else 0
        )

        team2_toss_percentage[index] = (
            sum(toss_team2) / len(toss_team2)
            if toss_team2 else 0
        )

    # --------------------------------------
    # SECOND: UPDATE HISTORIES
    # Only after all matches on this date
    # --------------------------------------

    for _, match in daily_matches.iterrows():

        team1 = match["team1"]
        team2 = match["team2"]
        winner = match["winner"]
        venue = match["venue"]
        toss_winner = match["toss_winner"]

        result1 = 1 if winner == team1 else 0
        result2 = 1 if winner == team2 else 0

        # UPDATE TEAM HISTORY

        team_history.setdefault(team1, []).append(result1)
        team_history.setdefault(team2, []).append(result2)

        # UPDATE HEAD-TO-HEAD HISTORY

        pair = tuple(sorted([team1, team2]))

        head_to_head.setdefault(pair, []).append(winner)

        # UPDATE VENUE HISTORY

        venue_history.setdefault(
            (team1, venue), []
        ).append(result1)

        venue_history.setdefault(
            (team2, venue), []
        ).append(result2)

        # UPDATE TOSS HISTORY
        # Only record the result for the toss winner

        if toss_winner == team1:
            toss_history.setdefault(team1, []).append(result1)

        elif toss_winner == team2:
            toss_history.setdefault(team2, []).append(result2)


# ==========================================
# 5. ADD FEATURES TO DATASET
# ==========================================

df["team1_recent_form"] = team1_form
df["team2_recent_form"] = team2_form

df["team1_win_percentage"] = team1_win_percentage
df["team2_win_percentage"] = team2_win_percentage

df["team1_h2h_percentage"] = team1_h2h
df["team2_h2h_percentage"] = team2_h2h

df["team1_venue_win_percentage"] = team1_venue_percentage
df["team2_venue_win_percentage"] = team2_venue_percentage

df["team1_toss_win_percentage"] = team1_toss_percentage
df["team2_toss_win_percentage"] = team2_toss_percentage


# ==========================================
# 6. SAVE DATASET
# ==========================================

output_path = "data/processed/t20_matches_features.csv"

df.to_csv(output_path, index=False)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

feature_columns = [
    "team1_recent_form",
    "team2_recent_form",
    "team1_win_percentage",
    "team2_win_percentage",
    "team1_h2h_percentage",
    "team2_h2h_percentage",
    "team1_venue_win_percentage",
    "team2_venue_win_percentage",
    "team1_toss_win_percentage",
    "team2_toss_win_percentage"
]

print("\n========== FEATURE ENGINEERING ==========")

print("\nDataset preview:")

print(df[[
    "date",
    "team1",
    "team2",
    "winner",
    "toss_winner",
    *feature_columns
]].head(15))

print("\n========== FEATURE STATISTICS ==========")

print(df[feature_columns].describe())

print("\n========== DATASET INFORMATION ==========")

print("Total matches:", len(df))
print("Total columns:", len(df.columns))
print("Dataset shape:", df.shape)

print("\nNew features:")

for column in feature_columns:
    print("-", column)

print("\nSaved successfully!")
print("Location:", output_path)
