from pathlib import Path
import pandas as pd

# HDFS log file
log_file = Path("data/logs/HDFS_2k.log")

print("================================")
print("HDFS FEATURE EXTRACTION")
print("================================")

if not log_file.exists():
    print("ERROR: HDFS_2k.log was not found.")
    exit()

# Read log file
with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

parsed_logs = []

for line in lines:
    line = line.strip()

    if not line:
        continue

    parts = line.split(" ", 4)

    if len(parts) < 5:
        continue

    date = parts[0]
    time = parts[1]
    log_id = parts[2]
    level = parts[3]
    remaining = parts[4]

    if ": " in remaining:
        component, message = remaining.split(": ", 1)
    else:
        component = remaining
        message = ""

    parsed_logs.append({
        "date": date,
        "time": time,
        "log_id": log_id,
        "level": level,
        "component": component,
        "message": message
    })

# Create DataFrame
df = pd.DataFrame(parsed_logs)

# Convert log ID to numeric
df["log_id_numeric"] = pd.to_numeric(
    df["log_id"],
    errors="coerce"
).fillna(0)

# Extract time features
df["hour"] = pd.to_numeric(
    df["time"].str[0:2],
    errors="coerce"
).fillna(0)

df["minute"] = pd.to_numeric(
    df["time"].str[2:4],
    errors="coerce"
).fillna(0)

df["second"] = pd.to_numeric(
    df["time"].str[4:6],
    errors="coerce"
).fillna(0)

# Extract text-based numerical features
df["message_length"] = df["message"].str.len()

df["word_count"] = df["message"].str.split().str.len()

df["component_length"] = df["component"].str.len()

# Select ML features
features = [
    "log_id_numeric",
    "hour",
    "minute",
    "second",
    "message_length",
    "word_count",
    "component_length"
]

X = df[features].fillna(0)

print("\nTotal logs:", len(df))

print("\nFeatures used for Machine Learning:")
print(features)

print("\nFeature Data:")
print(X.head(10).to_string(index=False))

print("\nFeature shape:", X.shape)