
import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/model_ready.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

# Select features and target
feature_columns = [
    "recent_form_difference",
    "win_percentage_difference",
    "h2h_difference",
    "venue_difference",
    "toss_performance_difference",
    "team1_won_toss",
    "toss_decision_bat"
]

# Find split using unique dates
unique_dates = sorted(df["date"].unique())
split_date_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_date_index]

# Keep every match on the same date together
train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["team1_won"]
y_test = test_df["team1_won"]

# Save datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Display results
print("\n========== CHRONOLOGICAL DATA SPLIT ==========")

print("Total matches:", len(df))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

print("\nSplit date:", split_date)

print("\nTraining date range:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nTesting date range:")
print(test_df["date"].min(), "to", test_df["date"].max())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nDatasets saved successfully!")
