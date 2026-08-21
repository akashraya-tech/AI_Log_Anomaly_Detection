from pathlib import Path
import re
import pandas as pd

# ========================================
# 1. Locate the HDFS log file
# ========================================

log_file = Path("data/logs/HDFS_2k.log")

print("================================")
print("HDFS EVENT TEMPLATE EXTRACTION")
print("================================")

if not log_file.exists():
    print("ERROR: HDFS_2k.log was not found.")
    exit()


# ========================================
# 2. Read the logs
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

print("Total logs:", len(df))


# ========================================
# 4. Create event template function
# ========================================

def create_template(message):

    template = message

    # Replace IP addresses
    template = re.sub(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "<IP>",
        template
    )

    # Replace HDFS block IDs
    template = re.sub(
        r"\bblk_-?\d+\b",
        "<BLOCK>",
        template
    )

    # Replace network ports
    template = re.sub(
        r":\d{4,5}\b",
        ":<PORT>",
        template
    )

    # Replace large integer values
    template = re.sub(
        r"\b\d{5,}\b",
        "<NUM>",
        template
    )

    # Replace smaller standalone numbers
    template = re.sub(
        r"\b\d+\b",
        "<NUM>",
        template
    )

    # Normalize multiple spaces
    template = re.sub(
        r"\s+",
        " ",
        template
    ).strip()

    return template


# ========================================
# 5. Generate templates
# ========================================

df["event_template"] = df["message"].apply(
    create_template
)


# ========================================
# 6. Count event templates
# ========================================

template_counts = (
    df["event_template"]
    .value_counts()
)


# ========================================
# 7. Display common templates
# ========================================

print("\n================================")
print("TOP 20 EVENT TEMPLATES")
print("================================")

print(
    template_counts.head(20).to_string()
)


# ========================================
# 8. Add frequency to each log
# ========================================

df["template_frequency"] = (
    df["event_template"]
    .map(template_counts)
)


# ========================================
# 9. Display examples
# ========================================

print("\n================================")
print("RAW MESSAGE TO EVENT TEMPLATE")
print("================================")
for _, row in df.head(10).iterrows():

    print("\nRAW:")
    print(row["message"])

    print("TEMPLATE:")
    print(row["event_template"])


# ========================================
# 10. Save results
# ========================================

output_file = Path(
    "data/logs/event_templates.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nEvent template data saved to:")
print(output_file)
