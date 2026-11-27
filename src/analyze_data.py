
import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("data/processed/t20_matches.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print("\n========== DATASET OVERVIEW ==========")

print("Total matches:", len(df))
print("Total teams:", len(set(df["team1"]) | set(df["team2"])))
print("Total venues:", df["venue"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# 1. Most successful teams
print("\n========== TOP WINNING TEAMS ==========")

team_wins = df["winner"].value_counts()

print(team_wins.head(10))

team_wins.head(10).sort_values().plot(
    kind="barh",
    title="Top 10 T20I Teams by Wins",
    xlabel="Matches Won",
    ylabel="Team"
)

plt.tight_layout()
plt.show()


# 2. Most active teams
print("\n========== MOST ACTIVE TEAMS ==========")

team_appearances = pd.concat([
    df["team1"],
    df["team2"]
]).value_counts()

print(team_appearances.head(10))

team_appearances.head(10).sort_values().plot(
    kind="barh",
    title="Top 10 Teams by Match Appearances",
    xlabel="Matches Played",
    ylabel="Team"
)

plt.tight_layout()
plt.show()


# 3. Toss winner vs match winner
print("\n========== TOSS IMPACT ==========")

toss_wins = (df["toss_winner"] == df["winner"]).sum()

toss_win_percentage = (
    toss_wins / len(df)
) * 100

print("Toss winner also won match:", toss_wins)
print("Percentage:", round(toss_win_percentage, 2), "%")


# 4. Toss decision analysis
print("\n========== TOSS DECISIONS ==========")

decision_stats = df.groupby("toss_decision").agg(
    matches=("winner", "count"),
    toss_winner_match_wins=(
        "winner",
        lambda winners: (
            winners == df.loc[winners.index, "toss_winner"]
        ).sum()
    )
)

decision_stats["win_percentage"] = (
    decision_stats["toss_winner_match_wins"]
    / decision_stats["matches"]
) * 100

print(decision_stats)


# 5. Most frequently used venues
print("\n========== TOP VENUES ==========")

venues = df["venue"].value_counts().head(10)

print(venues)

venues.sort_values().plot(
    kind="barh",
    title="Top 10 T20I Venues",
    xlabel="Matches",
    ylabel="Venue"
)

plt.tight_layout()
plt.show()


# 6. Matches by year
print("\n========== MATCHES BY YEAR ==========")

yearly_matches = df["date"].dt.year.value_counts().sort_index()

print(yearly_matches)

yearly_matches.plot(
    kind="line",
    marker="o",
    title="T20I Matches by Year",
    xlabel="Year",
    ylabel="Number of Matches"
)

plt.tight_layout()
plt.show()


print("\nEDA completed successfully!")
