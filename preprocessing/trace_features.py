from pathlib import Path
import pandas as pd

print("================================")
print("HDFS TRACE-LEVEL FEATURE EXTRACTION")
print("================================")

input_file = Path("data/logs/hdfs_traces.csv")

if not input_file.exists():
    print("ERROR: hdfs_traces.csv was not found.")
    exit()

# Load trace data
df = pd.read_csv(input_file)

df["message"] = df["message"].fillna("").astype(str)
df["event_template"] = df["event_template"].fillna("").astype(str)

print("\nTotal log events:", len(df))

# ========================================
# 1. Group logs by HDFS block
# ========================================

grouped = df.groupby("block_id")

# ========================================
# 2. Basic trace features
# ========================================

trace_features = grouped.agg(
    event_count=("block_id", "size"),
    unique_templates=("event_template", "nunique"),
    unique_components=("component", "nunique")
).reset_index()

# ========================================
# 3. Count IP addresses
# ========================================

df["ip_count"] = (
    df["message"]
    .str.count(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )
)

ip_counts = (
    df.groupby("block_id")["ip_count"]
    .sum()
    .reset_index(name="total_ip_count")
)

trace_features = trace_features.merge(
    ip_counts,
    on="block_id"
)

# ========================================
# 4. Event indicators
# ========================================

df["exception_event"] = (
    df["message"]
    .str.contains(
        "exception",
        case=False
    )
    .astype(int)
)

df["verification_event"] = (
    df["message"]
    .str.contains(
        "verification",
        case=False
    )
    .astype(int)
)

df["delete_event"] = (
    df["message"]
    .str.contains(
        "delete|deleting",
        case=False,
        regex=True
    )
    .astype(int)
)

df["allocate_event"] = (
    df["message"]
    .str.contains(
        "allocateBlock",
        case=False
    )
    .astype(int)
)

# ========================================
# 5. Aggregate event indicators
# ========================================

event_counts = (
    df.groupby("block_id")[
        [
            "exception_event",
            "verification_event",
            "delete_event",
            "allocate_event"
        ]
    ]
    .sum()
    .reset_index()
)

trace_features = trace_features.merge(
    event_counts,
    on="block_id"
)

# ========================================
# 6. Calculate trace duration
# ========================================

# Convert HHMMSS string to total seconds from midnight
def time_to_seconds(t_str):
    s = str(t_str).zfill(6)
    try:
        hh = int(s[0:2])
        mm = int(s[2:4])
        ss = int(s[4:6])
        return hh * 3600 + mm * 60 + ss
    except (ValueError, IndexError):
        return 0

df["time_seconds"] = df["time"].apply(time_to_seconds)

time_stats = (
    df.groupby("block_id")["time_seconds"]
    .agg(
        first_time="min",
        last_time="max"
    )
    .reset_index()
)

time_stats["trace_time_difference"] = (
    time_stats["last_time"]
    - time_stats["first_time"]
)

trace_features = trace_features.merge(
    time_stats[
        [
            "block_id",
            "trace_time_difference"
        ]
    ],
    on="block_id"
)

# ========================================
# 7. Display results
# ========================================

print("\nUnique HDFS traces:")

print(len(trace_features))

print("\nTrace feature columns:")

print(
    trace_features.columns.tolist()
)

print("\nFirst 20 traces:")

print(
    trace_features.head(20)
    .to_string(index=False)
)

# ========================================
# 8. Save trace features
# ========================================

output_file = Path(
    "data/logs/trace_features.csv"
)

trace_features.to_csv(
    output_file,
    index=False
)

print("\nTrace features saved to:")
print(output_file)