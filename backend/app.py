from flask import Flask, request, jsonify, render_template
from psycopg2.extras import RealDictCursor
from flask import session, redirect, url_for
from flask_cors import CORS
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
import joblib
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

# ========================================
# Flask Application
# ========================================
from dotenv import load_dotenv
load_dotenv()
app = Flask(
    __name__,
    template_folder="../templates"
)

app.secret_key = "akash_project_secret"
# Login counter
login_count = 0

app.permanent_session_lifetime = timedelta(minutes=1)

CORS(app)

# ========================================
# Load AI Model
# ========================================

# ========================================
# Load AI Model
# ========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "model"
    / "saved_model"
    / "isolation_forest_v3.joblib"
)

print("Model path:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("AKASH TEST 999")

# ========================================
# Database Connection
# ========================================
import os
import psycopg2

def get_db_connection():
    print("DB_PASSWORD =", os.getenv("DB_PASSWORD"))
    return psycopg2.connect(    
        host="pg-2cec95ed-ailoganomalydetection.a.aivencloud.com",
        database="defaultdb",
        user="avnadmin",
        password=os.getenv("DB_PASSWORD"),
        port="19268",
        sslmode="require",
    )
print("PostgreSQL database connection ready!")

# ========================================
# Feature Extraction
# ========================================

def extract_features(log_line):

    parts = log_line.strip().split(" ", 4)

    if len(parts) < 5:
        raise ValueError(
            "Invalid HDFS log format."
        )

    date = parts[0]
    time = parts[1]
    log_id = parts[2]
    level = parts[3]
    remaining = parts[4]

    if ": " in remaining:
        component, message = remaining.split(
            ": ", 1
        )
    else:
        component = remaining
        message = ""

    template = message

    template = re.sub(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "<IP>",
        template
    )

    template = re.sub(
        r"\bblk_-?\d+\b",
        "<BLOCK>",
        template
    )

    template = re.sub(
        r":\d{4,5}\b",
        ":<PORT>",
        template
    )

    template = re.sub(
        r"\b\d{5,}\b",
        "<NUM>",
        template
    )

    template = re.sub(
        r"\b\d+\b",
        "<NUM>",
        template
    )

    template = re.sub(
        r"\s+",
        " ",
        template
    ).strip()

    training_file =(
    BASE_DIR.parent
    / "data"
    / "logs"
    / "event_templates.csv"
   )
    

    training_df = pd.read_csv(
        training_file
    )

    template_frequency = (
        training_df["event_template"]
        .value_counts()
    )

    template_freq = template_frequency.get(
        template,
        0
    )

    component_frequency = (
        training_df["component"]
        .value_counts()
    )

    component_freq = component_frequency.get(
        component,
        0
    )

    message_length = len(message)

    word_count = len(
        message.split()
    )

    block_count = len(
        re.findall(
            r"\bblk_-?\d+\b",
            message
        )
    )

    ip_count = len(
        re.findall(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            message
        )
    )

    exception_present = int(
        "exception" in message.lower()
    )

    verification_present = int(
        "verification" in message.lower()
    )

    delete_present = int(
        bool(
            re.search(
                r"delete|deleting",
                message,
                re.IGNORECASE
            )
        )
    )

    allocate_present = int(
        "allocateblock" in message.lower()
    )

    hour = int(time[0:2])
    minute = int(time[2:4])

    features = pd.DataFrame([{

        "hour": hour,
        "minute": minute,
        "message_length": message_length,
        "word_count": word_count,
        "block_count": block_count,
        "ip_count": ip_count,
        "template_frequency": template_freq,
        "component_frequency": component_freq,
        "exception_present": exception_present,
        "verification_present": verification_present,
        "delete_present": delete_present,
        "allocate_present": allocate_present

    }])

    return features, {

        "date": date,
        "time": time,
        "level": level,
        "component": component,
        "message": message,
        "event_template": template

    }
    

# ========================================
# Home API
# ========================================

@app.route("/")
def dashboard():

    if not session.get("logged_in"):
        return redirect("/login")

    return render_template("dashboard.html")
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")
@app.route("/login-stats")
def login_stats():

    return jsonify({
        "total_logins": login_count
    })
@app.route("/test")
def test():
    return "Server Working"
# ========================================
# Prediction API
# ========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "log" not in data:
            return jsonify({
                "error": "Please provide a log field."
            }), 400

        log_line = data["log"]

        features, log_info = extract_features(log_line)

        print("\n===== FEATURES =====")
        print(features.to_string())
        print("====================\n")

        prediction = model.predict(features)[0]

        anomaly_score = model.decision_function(features)[0]

        if prediction == -1:
            status = "ANOMALY"
        else:
            status = "NORMAL"

        db = get_db_connection()
        cursor = db.cursor()

        sql = """
        INSERT INTO log_predictions
        (
            log_date,
            log_time,
            level,
            component,
            message,
            event_template,
            anomaly_status,
            anomaly_score
        )
        VALUES
        (%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            log_info["date"],
            log_info["time"],
            log_info["level"],
            log_info["component"],
            log_info["message"],
            log_info["event_template"],
            status,
            float(anomaly_score)
        )

        cursor.execute(sql, values)

        db.commit()

        cursor.close()
        db.close()

        return jsonify({
            "status": status,
            "anomaly_score": round(float(anomaly_score), 6),
            "log": log_info
        })

    except Exception as e:

        import traceback

        print("\n===== ERROR =====")
        traceback.print_exc()
        print("=================\n")

        return jsonify({
            "error": str(e)
        }), 500
# ========================================
# Dashboard Stats API
# ========================================

@app.route("/stats")
def get_stats():

    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute("SELECT COUNT(*) FROM log_predictions")
    total_logs = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM log_predictions
        WHERE anomaly_status='ANOMALY'
    """)
    anomalies = cursor.fetchone()[0]

    normal_logs = total_logs - anomalies

    cursor.close()
    db.close()

    return jsonify({
        "total_logs": total_logs,
        "anomalies": anomalies,
        "normal_logs": normal_logs
    })
# ========================================
# Dashboard Logs API
# ========================================

@app.route("/logs", methods=["GET"])
def get_logs():

    db = get_db_connection()
    cursor = db.cursor(cursor_factory=RealDictCursor)

    cursor.execute(
        """
        SELECT
            id,
            component,
            anomaly_status,
            anomaly_score,
            level,
            message,
            log_date
        FROM log_predictions
        ORDER BY id DESC
        LIMIT 20
        """
    )

    logs = cursor.fetchall()

    cursor.close()
    db.close()

    return jsonify(logs)
# ========================================
# Dashboard login 
# ========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    global login_count

    error = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "AILog@2026#Secure":

            login_count += 1

            session.permanent = True
            session["logged_in"] = True

            return redirect("/")

        else:

            error = "Wrong username or password"

    return render_template(
        "login.html",
        error=error
    )
# ========================================
# Start Server
# ========================================

if __name__ == "__main__":
  
    print("================================")
    print("AI LOG ANOMALY DETECTION SERVER")
    print("================================")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
