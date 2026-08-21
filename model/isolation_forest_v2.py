from pathlib import Path
import re
import pandas as pd
from sklearn.ensemble import IsolationForest

# ========================================
# 1. Locate log file
# ========================================

log_file = Path("data/logs/HDFS_2k.log")

print("================================")
print("ISOLATION FOREST V2")
print("IMPROVED FEATURE ENGINEERING")
print("================================")

if not log_file.exists():
    print("ERROR: HDFS_2k.log was not found.")
    exit()

# ========================================
# 2. Read logs
# ========================================

with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

# ========================================
# 3. Parse logs
# ========================================

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

df = pd.DataFrame(parsed_logs)

print("\nTotal parsed logs:", len(df))


# ========================================
# 4. Extract time features
# ========================================

df["hour"] = pd.to_numeric(
    df["time"].str[0:2],
    errors="coerce"
).fillna(0)

df["minute"] = pd.to_numeric(
    df["time"].str[2:4],
    errors="coerce"
).fillna(0)


# ========================================
# 5. Basic message features
# ========================================

df["message_length"] = df["message"].str.len()

df["word_count"] = (
    df["message"]
    .str.split()
    .str.len()
)


# ========================================
# 6. Count IP addresses
# ========================================

ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

df["ip_count"] = (
    df["message"]
    .str.count(ip_pattern)
)


# ========================================
# 7. Count HDFS block IDs
# ========================================

block_pattern = r"\bblk_-?\d+\b"

df["block_count"] = (
    df["message"]
    .str.count(block_pattern)
)


# ========================================
# 8. Detect filesystem paths
# ========================================

df["path_present"] = (
    df["message"]
    .str.contains(r"/user/|/tmp/|/data/", regex=True)
    .astype(int)
)


# ========================================
# 9. Detect network ports
# ========================================

df["port_present"] = (
    df["message"]
    .str.contains(r":\d{4,5}\b", regex=True)
    .astype(int)
)


# ========================================
# 10. Identify event type
# ========================================

def identify_event(message):

    message_lower = message.lower()

    if "packetresponder" in message_lower:
        return "PacketResponder"

    elif "received block" in message_lower:
        return "ReceivedBlock"

    elif "allocateblock" in message_lower:
        return "AllocateBlock"

    elif "addstoredblock" in message_lower:
        return "AddStoredBlock"

    elif "delete" in message_lower and "blk_" in message_lower:
        return "DeleteBlock"

    elif "verification succeeded" in message_lower:
        return "BlockVerification"

    elif "receiving block" in message_lower:
        return "ReceivingBlock"

    else:
        return "Other"


df["event_type"] = df["message"].apply(identify_event)


# ========================================
# 11. Calculate event frequencies
# ========================================

event_frequency = (
    df["event_type"]
    .value_counts()
)

df["event_frequency"] = (
    df["event_type"]
    .map(event_frequency)
)


# ========================================
# 12. Calculate component frequencies
# ========================================

component_frequency = (
    df["component"]
    .value_counts()
)

df["component_frequency"] = (
    df["component"]
    .map(component_frequency)
)


# ========================================
# 13. One-hot encode event type
# ========================================

event_encoded = pd.get_dummies(
    df["event_type"],
    prefix="event",
    dtype=int
)


# ========================================
# 14. Select numerical features
# ========================================

base_features = [
    "hour",
    "minute",
    "message_length",
    "word_count",
    "ip_count",
    "block_count",
    "path_present",
    "port_present",
    "event_frequency",
    "component_frequency"
]

X = pd.concat(
    [
        df[base_features],
        event_encoded
    ],
    axis=1
)

X = X.fillna(0)


# ========================================
# 15. Display feature information
# ========================================

print("\nEvent types found:")

print(
    df["event_type"]
    .value_counts()
    .to_string()
)

print("\nFeatures used by the improved model:")

print(X.columns.tolist())

print("\nFeature matrix shape:")

print(X.shape)


# ========================================
# 16. Create Isolation Forest
# ========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


# ========================================
# 17. Train model
# ========================================

model.fit(X)


# ========================================
# 18. Predict anomalies
# ========================================

df["prediction"] = model.predict(X)

df["anomaly_score"] = model.decision_function(X)

df["anomaly"] = df["prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})


# ========================================
# 19. Summary
# ========================================

total_logs = len(df)

normal_logs = (
    df["anomaly"] == "NORMAL"
).sum()

anomaly_logs = (
    df["anomaly"] == "ANOMALY"
).sum()

print("\n================================")
print("MODEL RESULTS")
print("================================")

print("Total logs:", total_logs)
print("Normal logs:", normal_logs)
print("Anomalous logs:", anomaly_logs)


# ========================================
# 20. Most unusual logs
# ========================================

print("\n================================")
print("TOP 20 MOST UNUSUAL LOGS")
print("================================")

most_unusual = (
    df.sort_values("anomaly_score")
    .head(20)
)

print(
    most_unusual[
        [
            "date",
            "time",
            "component",
            "event_type",
            "message_length",
            "block_count",
            "anomaly",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# ========================================
# 21. Save improved results
# ========================================

output_file = Path(
    "data/logs/anomaly_results_v2.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nImproved results saved to:")

print(output_file)
# ========================================
# 22. Analyze anomaly event types
# ========================================

print("\n================================")
print("ANOMALY EVENT TYPE DISTRIBUTION")
print("================================")

anomaly_event_counts = (
    df[df["anomaly"] == "ANOMALY"]["event_type"]
    .value_counts()
)

print(anomaly_event_counts.to_string())


# ========================================
# 23. Analyze normal event types
# ========================================

print("\n================================")
print("NORMAL EVENT TYPE DISTRIBUTION")
print("================================")

normal_event_counts = (
    df[df["anomaly"] == "NORMAL"]["event_type"]
    .value_counts()
)

print(normal_event_counts.to_string())