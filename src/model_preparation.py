
import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/processed/t20_matches_features.csv")

# Create comparative features
df["recent_form_difference"] = (
    df["team1_recent_form"] - df["team2_recent_form"]
)

df["win_percentage_difference"] = (
    df["team1_win_percentage"] - df["team2_win_percentage"]
)

df["h2h_difference"] = (
    df["team1_h2h_percentage"] - df["team2_h2h_percentage"]
)

df["venue_difference"] = (
    df["team1_venue_win_percentage"]
    - df["team2_venue_win_percentage"]
)

df["toss_performance_difference"] = (
    df["team1_toss_win_percentage"]
    - df["team2_toss_win_percentage"]
)

# Encode actual toss information
# Symmetrical toss features
df["toss_advantage"] = df.apply(
    lambda row: 1 if row["toss_winner"] == row["team1"] else -1,
    axis=1
)

df["toss_decision_advantage"] = (
    df["toss_advantage"] *
    df["toss_decision"].map({"bat": 1, "field": -1})
)

# Select model features
feature_columns = [
    "recent_form_difference",
    "win_percentage_difference",
    "h2h_difference",
    "venue_difference",
    "toss_performance_difference",
    "toss_advantage",
    "toss_decision_advantage"
]

X = df[feature_columns]
y = df["team1_won"]

# Create model-ready dataset
model_df = pd.concat(
    [df[["date", "match_id", "team1", "team2"]], X, y],
    axis=1
)

model_df.to_csv(
    "data/processed/model_ready.csv",
    index=False
)

print("\n========== UPDATED MODEL PREPARATION ==========")
print("Total matches:", len(df))
print("Total features:", len(feature_columns))
print("\nFeatures:")
print(X.head())

print("\nMissing values:")
print(X.isnull().sum())

print("\nDataset shape:", model_df.shape)
print("\nSaved: data/processed/model_ready.csv")
