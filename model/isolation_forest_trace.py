from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest

print("================================")
print("HDFS TRACE-LEVEL ANOMALY DETECTION")
print("================================")

# ========================================
# 1. Load trace features
# ========================================

input_file = Path(
    "data/logs/trace_features.csv"
)

if not input_file.exists():
    print("ERROR: trace_features.csv was not found.")
    exit()

df = pd.read_csv(input_file)

print("\nTotal HDFS traces:", len(df))


# ========================================
# 2. Select ML features
# ========================================

features = [
    "event_count",
    "unique_templates",
    "unique_components",
    "total_ip_count",
    "exception_event",
    "verification_event",
    "delete_event",
    "allocate_event",
    "trace_time_difference"
]

X = df[features].fillna(0)


# ========================================
# 3. Display feature information
# ========================================

print("\nFeatures used:")

for feature in features:
    print("-", feature)

print("\nFeature matrix shape:")

print(X.shape)


# ========================================
# 4. Create Isolation Forest
# ========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


# ========================================
# 5. Train
# ========================================

model.fit(X)


# ========================================
# 6. Predict
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
# 7. Results
# ========================================

normal_count = (
    df["anomaly"] == "NORMAL"
).sum()

anomaly_count = (
    df["anomaly"] == "ANOMALY"
).sum()

print("\n================================")
print("TRACE MODEL RESULTS")
print("================================")

print("Total traces:", len(df))
print("Normal traces:", normal_count)
print("Anomalous traces:", anomaly_count)


# ========================================
# 8. Most unusual traces
# ========================================

print("\n================================")
print("TOP 20 MOST UNUSUAL TRACES")
print("================================")

top_traces = (
    df.sort_values("anomaly_score")
    .head(20)
)

print(
    top_traces[
        [
            "block_id",
            "event_count",
            "unique_templates",
            "unique_components",
            "total_ip_count",
            "exception_event",
            "verification_event",
            "delete_event",
            "allocate_event",
            "anomaly",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# ========================================
# 9. Save results
# ========================================

output_file = Path(
    "data/logs/trace_anomaly_results.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nTrace anomaly results saved to:")
print(output_file)