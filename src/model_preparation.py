
import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/processed/t20_matches_features.csv")

df["date"] = pd.to_datetime(df["date"])

# ==========================================
# 2. CREATE COMPARATIVE FEATURES
# ==========================================

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

# ==========================================
# 3. SELECT MODEL FEATURES
# ==========================================

feature_columns = [
    "recent_form_difference",
    "win_percentage_difference",
    "h2h_difference",
    "venue_difference",
    "toss_performance_difference"
]

X = df[feature_columns]

y = df["team1_won"]

# ==========================================
# 4. CHECK DATA QUALITY
# ==========================================

print("\n========== MODEL PREPARATION ==========")

print("\nFeature columns:")
for column in feature_columns:
    print("-", column)

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:")
print(X.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ==========================================
# 5. SAVE MODEL-READY DATASET
# ==========================================

model_df = pd.concat(
    [df[["date", "match_id", "team1", "team2"]], X, y],
    axis=1
)

model_df.to_csv(
    "data/processed/model_ready.csv",
    index=False
)

print("\nDataset shape:", model_df.shape)

print("\nSaved successfully!")
print("Location: data/processed/model_ready.csv")
