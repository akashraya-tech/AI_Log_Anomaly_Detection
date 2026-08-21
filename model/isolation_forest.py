from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest

# ----------------------------------------
# 1. Locate the HDFS log file
# ----------------------------------------

log_file = Path("data/logs/HDFS_2k.log")

print("================================")
print("ISOLATION FOREST ANOMALY DETECTION")
print("================================")

if not log_file.exists():
    print("ERROR: HDFS_2k.log was not found.")
    exit()

# ----------------------------------------
# 2. Read the log file
# ----------------------------------------

with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

# ----------------------------------------
# 3. Parse the logs
# ----------------------------------------

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

# ----------------------------------------
# 4. Create DataFrame
# ----------------------------------------

df = pd.DataFrame(parsed_logs)

# ----------------------------------------
# 5. Feature extraction
# ----------------------------------------

df["log_id_numeric"] = pd.to_numeric(
    df["log_id"],
    errors="coerce"
).fillna(0)

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

df["message_length"] = df["message"].str.len()

df["word_count"] = df["message"].str.split().str.len()

df["component_length"] = df["component"].str.len()

# ----------------------------------------
# 6. Select ML features
# ----------------------------------------

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

# ----------------------------------------
# 7. Create Isolation Forest
# ----------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# ----------------------------------------
# 8. Train the model
# ----------------------------------------

model.fit(X)

# ----------------------------------------
# 9. Predict anomalies
# ----------------------------------------

# ----------------------------------------
# 9. Predict anomalies and calculate scores
# ----------------------------------------

df["prediction"] = model.predict(X)

# Lower score = more unusual
df["anomaly_score"] = model.decision_function(X)

# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["anomaly"] = df["prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

# ----------------------------------------
# 10. Display results
# ----------------------------------------

# ----------------------------------------
# 10. Display results
# ----------------------------------------

total_logs = len(df)

normal_logs = (df["anomaly"] == "NORMAL").sum()

anomaly_logs = (df["anomaly"] == "ANOMALY").sum()

print("\nModel training completed successfully!")

print("\nTotal logs:", total_logs)
print("Normal logs:", normal_logs)
print("Anomalous logs:", anomaly_logs)

print("\nFirst 20 detection results:\n")

print(
    df[
        [
            "date",
            "time",
            "level",
            "component",
            "message",
            "anomaly",
            "anomaly_score"
        ]
    ].head(20).to_string(index=False)
)

# ----------------------------------------
# 11. Show the most unusual logs
# ----------------------------------------

print("\n================================")
print("TOP 20 MOST UNUSUAL LOGS")
print("================================")

most_unusual = df.sort_values(
    by="anomaly_score"
).head(20)

print(
    most_unusual[
        [
            "date",
            "time",
            "level",
            "component",
            "message",
            "anomaly",
            "anomaly_score"
        ]
    ].to_string(index=False)
)

# ----------------------------------------
# 12. Save detection results
# ----------------------------------------

output_file = Path("data/logs/anomaly_results.csv")

df.to_csv(output_file, index=False)

print("\nResults saved to:")
print(output_file)