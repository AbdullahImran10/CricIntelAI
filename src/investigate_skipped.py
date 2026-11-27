
import os
import json
from collections import Counter

DATA_PATH = "data/raw"

reasons = Counter()
examples = {}

for filename in os.listdir(DATA_PATH):

    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(DATA_PATH, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    info = data.get("info", {})
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")

    reason = None

    if len(teams) != 2:
        reason = "Not exactly two teams"

    elif winner not in teams:
        if winner is None:
            reason = "No winner recorded"
        else:
            reason = "Winner not among listed teams"

    if reason:
        reasons[reason] += 1
        examples.setdefault(reason, []).append(filename)

print("\n========== SKIPPED MATCH INVESTIGATION ==========")

for reason, count in reasons.items():
    print(f"\n{reason}: {count}")
    print("Example files:", examples[reason][:5])

print("\nTotal skipped matches:", sum(reasons.values()))

