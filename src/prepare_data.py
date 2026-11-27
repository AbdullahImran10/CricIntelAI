
import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/t20_matches.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Create binary target
df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

# Display results
print("\n========== TARGET COLUMN ==========")

print(df[["team1", "team2", "winner", "team1_won"]].head(10))

print("\nTarget distribution:")
print(df["team1_won"].value_counts())

print("\nTarget percentages:")
print(df["team1_won"].value_counts(normalize=True) * 100)

# Save updated dataset
df.to_csv("data/processed/t20_matches_prepared.csv", index=False)

print("\nPrepared dataset saved successfully!")
