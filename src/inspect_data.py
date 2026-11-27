
import os
import json

DATA_PATH = "data/raw"

# Get all JSON files
files = [
    file for file in os.listdir(DATA_PATH)
    if file.endswith(".json")
]

print("Total JSON files:", len(files))

# Open the first match
first_file = files[0]
file_path = os.path.join(DATA_PATH, first_file)

with open(file_path, "r", encoding="utf-8") as f:
    match_data = json.load(f)

print("\nFirst match filename:", first_file)

print("\nAvailable top-level information:")
for key in match_data:
    print("-", key)

print("\nMatch information:")
print(json.dumps(match_data.get("info", {}), indent=4))
