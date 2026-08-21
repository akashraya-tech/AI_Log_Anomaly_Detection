from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
# ========================================
# 1. Locate event-template data
# ========================================

input_file = Path("data/logs/event_templates.csv")

print("================================")
print("ISOLATION FOREST V3")
print("TEMPLATE-BASED FEATURES")
print("================================")

if not input_file.exists():
    print("ERROR: event_templates.csv was not found.")
    exit()


# ========================================
# 2. Load structured template data
# ========================================

df = pd.read_csv(input_file)

print("\nTotal logs:", len(df))


# ========================================
# 3. Calculate template frequency
# ========================================

template_frequency = (
    df["event_template"]
    .value_counts()
)

df["template_frequency"] = (
    df["event_template"]
    .map(template_frequency)
)


# ========================================
# 4. Calculate component frequency
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
# 5. Extract numerical features
# ========================================

df["message_length"] = (
    df["message"]
    .fillna("")
    .str.len()
)

df["word_count"] = (
    df["message"]
    .fillna("")
    .str.split()
    .str.len()
)

df["block_count"] = (
    df["message"]
    .fillna("")
    .str.count(r"\bblk_-?\d+\b")
)

df["ip_count"] = (
    df["message"]
    .fillna("")
    .str.count(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )
)


# ========================================
# 6. Event indicators
# ========================================

df["exception_present"] = (
    df["message"]
    .fillna("")
    .str.contains(
        "exception",
        case=False
    )
    .astype(int)
)

df["verification_present"] = (
    df["message"]
    .fillna("")
    .str.contains(
        "verification",
        case=False
    )
    .astype(int)
)

df["delete_present"] = (
    df["message"]
    .fillna("")
    .str.contains(
        "delete|deleting",
        case=False,
        regex=True
    )
    .astype(int)
)

df["allocate_present"] = (
    df["message"]
    .fillna("")
    .str.contains(
        "allocateBlock",
        case=False
    )
    .astype(int)
)


# ========================================
# 7. Time features
# ========================================

df["hour"] = pd.to_numeric(
    df["time"].astype(str).str[0:2],
    errors="coerce"
).fillna(0)

df["minute"] = pd.to_numeric(
    df["time"].astype(str).str[2:4],
    errors="coerce"
).fillna(0)


# ========================================
# 8. Select ML features
# ========================================

features = [
    "hour",
    "minute",
    "message_length",
    "word_count",
    "block_count",
    "ip_count",
    "template_frequency",
    "component_frequency",
    "exception_present",
    "verification_present",
    "delete_present",
    "allocate_present"
]

X = df[features].fillna(0)


# ========================================
# 9. Display features
# ========================================

print("\nFeatures used:")

for feature in features:
    print("-", feature)

print("\nFeature matrix shape:")

print(X.shape)


# ========================================
# 10. Create Isolation Forest
# ========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


# ========================================
# 11. Train model
# ========================================

model.fit(X)
# ========================================
# Save trained model
# ========================================

model_directory = Path("model/saved_model")

model_directory.mkdir(
    parents=True,
    exist_ok=True
)

model_file = (
    model_directory /
    "isolation_forest_v3.joblib"
)

joblib.dump(
    model,
    model_file
)

print("\nTrained model saved to:")
print(model_file)


# ========================================
# 12. Predict anomalies
# ========================================

df["prediction"] = model.predict(X)

df["anomaly_score"] = (
    model.decision_function(X)
)

df["anomaly"] = df["prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})


# ========================================
# 13. Display model results
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
# 14. Show top anomalies
# ========================================

print("\n================================")
print("TOP 20 MOST UNUSUAL LOGS")
print("================================")

top_anomalies = (
    df.sort_values("anomaly_score")
    .head(20)
)

print(
    top_anomalies[
        [
            "time",
            "component",
            "event_template",
            "exception_present",
            "verification_present",
            "delete_present",
            "anomaly",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# ========================================
# 15. Save V3 results
# ========================================

output_file = Path(
    "data/logs/anomaly_results_v3.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nV3 results saved to:")
print(output_file)