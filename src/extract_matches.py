
import os
import json
import pandas as pd

# Dataset directory
DATA_PATH = "data/raw"

# Store extracted matches
matches = []

# Validation counters
total_files = 0
skipped_matches = 0

# Read every JSON file
for filename in os.listdir(DATA_PATH):

    if not filename.endswith(".json"):
        continue

    total_files += 1

    file_path = os.path.join(DATA_PATH, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    info = data.get("info", {})

    # Extract match information
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})

    # Check for exactly two participating teams
    if len(teams) != 2:
        skipped_matches += 1
        continue

    # Extract winner
    winner = outcome.get("winner")

    # Skip matches without a valid winner
    if winner not in teams:
        skipped_matches += 1
        continue

    # Extract match details
    match = {
        "match_id": filename.replace(".json", ""),
        "date": info.get("dates", [None])[0],
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner
    }

    matches.append(match)


# Convert records into DataFrame
df = pd.DataFrame(matches)

# Create processed directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Save dataset as CSV
output_path = "data/processed/t20_matches.csv"

df.to_csv(output_path, index=False)


# Display dataset information
print("\n========== DATASET EXTRACTION ==========")

print("Total JSON files:", total_files)
print("Valid matches:", len(df))
print("Skipped matches:", skipped_matches)

print("\n========== DATASET VALIDATION ==========")

# Check duplicate match IDs
duplicates = df["match_id"].duplicated().sum()

print("Duplicate match IDs:", duplicates)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display dataset shape
print("\nDataset shape:", df.shape)

# Display first five matches
print("\nDataset preview:")
print(df.head())

# Display winner distribution
print("\nMatch winner distribution:")
print(df["winner"].value_counts().head(10))

print("\n========================================")
print("Dataset saved successfully!")
print("Location:", output_path)
