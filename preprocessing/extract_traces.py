from pathlib import Path
import re
import pandas as pd

print("================================")
print("HDFS TRACE EXTRACTION")
print("================================")

input_file = Path("data/logs/event_templates.csv")

if not input_file.exists():
    print("ERROR: event_templates.csv was not found.")
    exit()

# Load parsed log data
df = pd.read_csv(input_file)

# Make sure message is treated as text
df["message"] = df["message"].fillna("").astype(str)

# Extract HDFS block ID
block_pattern = r"\bblk_-?\d+\b"

df["block_id"] = df["message"].str.extract(
    f"({block_pattern})",
    expand=False
)

# Count logs with and without block IDs
with_block = df["block_id"].notna().sum()
without_block = df["block_id"].isna().sum()

print("\nTotal logs:", len(df))
print("Logs containing block ID:", with_block)
print("Logs without block ID:", without_block)

# Keep only logs that have a block ID
trace_df = df.dropna(
    subset=["block_id"]
).copy()

# Count events per block
trace_sizes = (
    trace_df.groupby("block_id")
    .size()
    .sort_values(ascending=False)
)

print("\nUnique HDFS blocks:", trace_df["block_id"].nunique())

print("\nTop 20 blocks by number of log events:")

print(
    trace_sizes.head(20).to_string()
)

# Add event count for each block
trace_df["events_in_trace"] = (
    trace_df["block_id"]
    .map(trace_sizes)
)

# Save trace-level data
output_file = Path(
    "data/logs/hdfs_traces.csv"
)

trace_df.to_csv(
    output_file,
    index=False
)

print("\nTrace data saved to:")
print(output_file)