from pathlib import Path
import pandas as pd

# Location of the HDFS log file
log_file = Path("data/logs/HDFS_2k.log")

print("================================")
print("HDFS LOG PARSER")
print("================================")

if not log_file.exists():
    print("ERROR: HDFS_2k.log was not found.")
    exit()

# Read all log lines
with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

parsed_logs = []

for line in lines:
    line = line.strip()

    # Ignore empty lines
    if not line:
        continue

    # Split only the first 4 spaces
    parts = line.split(" ", 4)

    if len(parts) < 5:
        continue

    date = parts[0]
    time = parts[1]
    log_id = parts[2]
    level = parts[3]
    remaining = parts[4]

    # Separate component and message
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

# Convert to Pandas DataFrame
df = pd.DataFrame(parsed_logs)

print("\nTotal parsed logs:", len(df))

print("\nStructured log data:")
print(df.head(10).to_string(index=False))

print("\nColumns:")
print(df.columns.tolist())