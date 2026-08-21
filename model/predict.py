from pathlib import Path
import joblib
import pandas as pd


# ========================================
# 1. Locate saved model
# ========================================

model_file = Path(
    "model/saved_model/isolation_forest_v3.joblib"
)

print("================================")
print("AI LOG ANOMALY PREDICTION")
print("================================")


# ========================================
# 2. Check model
# ========================================

if not model_file.exists():

    print("ERROR: Trained model was not found.")

    print("Expected location:")
    print(model_file)

    exit()


# ========================================
# 3. Load trained model
# ========================================

model = joblib.load(model_file)

print("\nTrained model loaded successfully!")


# ========================================
# 4. Load event-template data
# ========================================

input_file = Path(
    "data/logs/event_templates.csv"
)

if not input_file.exists():

    print("ERROR: event_templates.csv was not found.")

    exit()


df = pd.read_csv(input_file)

print("Log data loaded successfully!")

print("Total logs:", len(df))


# ========================================
# 5. Recreate features
# ========================================

df["message"] = (
    df["message"]
    .fillna("")
    .astype(str)
)

df["event_template"] = (
    df["event_template"]
    .fillna("")
    .astype(str)
)


# Template frequency

template_frequency = (
    df["event_template"]
    .value_counts()
)

df["template_frequency"] = (
    df["event_template"]
    .map(template_frequency)
)


# Component frequency

component_frequency = (
    df["component"]
    .value_counts()
)

df["component_frequency"] = (
    df["component"]
    .map(component_frequency)
)


# Message features

df["message_length"] = (
    df["message"].str.len()
)

df["word_count"] = (
    df["message"]
    .str.split()
    .str.len()
)


# Block count

df["block_count"] = (
    df["message"]
    .str.count(r"\bblk_-?\d+\b")
)


# IP count

df["ip_count"] = (
    df["message"]
    .str.count(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )
)


# ========================================
# 6. Event indicators
# ========================================

df["exception_present"] = (
    df["message"]
    .str.contains(
        "exception",
        case=False
    )
    .astype(int)
)

df["verification_present"] = (
    df["message"]
    .str.contains(
        "verification",
        case=False
    )
    .astype(int)
)

df["delete_present"] = (
    df["message"]
    .str.contains(
        "delete|deleting",
        case=False,
        regex=True
    )
    .astype(int)
)

df["allocate_present"] = (
    df["message"]
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
    df["time"]
    .astype(str)
    .str[0:2],
    errors="coerce"
).fillna(0)

df["minute"] = pd.to_numeric(
    df["time"]
    .astype(str)
    .str[2:4],
    errors="coerce"
).fillna(0)


# ========================================
# 8. Select features
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
# 9. Make predictions
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
# 10. Display summary
# ========================================

normal_count = (
    df["anomaly"] == "NORMAL"
).sum()

anomaly_count = (
    df["anomaly"] == "ANOMALY"
).sum()


print("\n================================")
print("PREDICTION RESULTS")
print("================================")

print("Total logs:", len(df))

print("Normal logs:", normal_count)

print("Anomalous logs:", anomaly_count)


# ========================================
# 11. Display sample predictions
# ========================================

print("\n================================")
print("SAMPLE PREDICTIONS")
print("================================")

print(

    df[
        [
            "time",
            "component",
            "event_template",
            "anomaly",
            "anomaly_score"
        ]
    ]
    .head(20)
    .to_string(index=False)

)


# ========================================
# 12. Save predictions
# ========================================

output_file = Path(
    "data/logs/prediction_results.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nPrediction results saved to:")

print(output_file)