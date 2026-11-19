import zipfile
import io
import json
from pathlib import Path

import pandas as pd
from urllib.request import urlopen

# Downloading meta data for IPL matches from cricsheet.org and filtering for RCB matches

project_dir = Path(__file__).resolve().parent.parent
url = "https://cricsheet.org/downloads/ipl_json.zip"

with urlopen(url, timeout=120) as response:
	archive = zipfile.ZipFile(io.BytesIO(response.read()))

print("Total files:", len(archive.namelist()))

# Extract match features for RCB matches

match_records = []

for filename in archive.namelist():

    if not filename.endswith(".json"):
        continue

    match = json.loads(archive.read(filename))
    info = match["info"]

    match_id = filename.split("/")[-1].replace(".json", "")

    match_date = str(info["dates"][0])
    teams = info["teams"]

    rcb_names = [
        "Royal Challengers Bangalore",
        "Royal Challengers Bengaluru"
    ]

    if not any(team in rcb_names for team in teams):
        continue

    opponent = next(
        team for team in teams
        if team not in rcb_names
    )

    match_records.append({
        "match_id": match_id,
        "season": info.get("season"),
        "match_date": match_date,
        "opponent": opponent,
        "venue": info.get("venue")
    })

matches_df = pd.DataFrame(match_records)

print(matches_df.head())
print("RCB matches:", len(matches_df))

#  merge with KOhli delivery data

kohli_file = (
    project_dir
    / "Data"
    / "processed"
    / "KOhli_delivery_Combined_2009-2026.csv"
)
if not kohli_file.is_file():
    raise FileNotFoundError(f"Input file not found: {kohli_file}")

kohli_df = pd.read_csv(kohli_file, dtype={"match_id": "string"})

matches_df["match_id"] = (
    matches_df["match_id"].astype("string")
)

final_df = kohli_df.merge(
    matches_df,
    on="match_id",
    how="left",
    validate="many_to_one",
    indicator=True,
    suffixes=("", "_metadata")
)

print(final_df["_merge"].value_counts())

# check missing match metadata

unmatched = final_df[
    final_df["_merge"] == "left_only"
]

print("Unmatched deliveries:", len(unmatched))

print(
    "Unmatched matches:",
    unmatched["match_id"].nunique()
)

print(
    unmatched["match_id"].drop_duplicates().tolist()
)

#  save enriched data to CSV

final_df = final_df.drop(columns=["_merge"])

output_file = (
    project_dir
    / "Data"
    / "processed"
    / "kohli_ipl_with_match_metadata.csv"
)
final_df.to_csv(output_file, index=False)

print(f"Dataset saved successfully: {output_file}")

#  varify the saved CSV file
# Check seasons
print(final_df["season_metadata"].value_counts(dropna=False))

# Check missing dates
print(final_df["match_date"].isna().sum())

# Check row count
print("Original rows:", len(kohli_df))
print("Final rows:", len(final_df))

# Check unique matches
print("Matches:", final_df["match_id"].nunique())