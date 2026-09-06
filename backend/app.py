import os
import re
import secrets
from datetime import timedelta
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ========================================
# Flask Application
# ========================================
app = Flask(
    __name__,
    template_folder="../templates"
)

app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key_change_in_env")
app.permanent_session_lifetime = timedelta(minutes=30)
CORS(app)

login_count = 0

# ========================================
# Authentication Helpers & API Key Decorator
# ========================================
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "AILog@2026#Secure")
API_KEY = os.getenv("API_KEY", "logdetect-api-key-2026")

def generate_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(16)
    return session["_csrf_token"]

app.jinja_env.globals["csrf_token"] = generate_csrf_token

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow authenticated session
        if session.get("logged_in"):
            return f(*args, **kwargs)
        # Allow via API Key header
        client_key = request.headers.get("X-API-Key")
        if client_key and client_key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized access. Provide valid session or X-API-Key."}), 401
    return decorated_function

# ========================================
# Load AI Model
# ========================================
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
print("AI Model loaded successfully!")

# ========================================
# Pre-cache Training Frequencies (Performance Optimization)
# ========================================
TRAINING_FILE = (
    BASE_DIR.parent
    / "data"
    / "logs"
    / "event_templates.csv"
)

if TRAINING_FILE.exists():
    print("Pre-caching template and component frequencies...")
    _training_df = pd.read_csv(TRAINING_FILE)
    CACHED_TEMPLATE_FREQ = _training_df["event_template"].value_counts().to_dict()
    CACHED_COMPONENT_FREQ = _training_df["component"].value_counts().to_dict()
    print("Frequencies pre-cached successfully!")
else:
    print("Warning: event_templates.csv not found for pre-caching. Using empty dicts.")
    CACHED_TEMPLATE_FREQ = {}
    CACHED_COMPONENT_FREQ = {}

# ========================================
# Database Connection Pool
# ========================================
DB_HOST = os.getenv("DB_HOST", "pg-2cec95ed-ailoganomalydetection.a.aivencloud.com")
DB_PORT = os.getenv("DB_PORT", "19268")
DB_NAME = os.getenv("DB_NAME", "defaultdb")
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD")

db_pool = None
try:
    if DB_PASSWORD:
        db_pool = pool.SimpleConnectionPool(
            1,
            10,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode="require",
        )
        print("PostgreSQL connection pool initialized successfully!")
    else:
        print("DB_PASSWORD not configured. Database pooling not started.")
except Exception as e:
    print(f"Failed to initialize PostgreSQL pool: {e}")

def get_db_conn():
    if db_pool:
        return db_pool.getconn()
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
    )

def release_db_conn(conn):
    if db_pool and conn:
        db_pool.putconn(conn)
    elif conn:
        conn.close()

# ========================================
# Feature Extraction
# ========================================
def extract_features(log_line):
    parts = log_line.strip().split(" ", 4)

    if len(parts) < 5:
        raise ValueError("Invalid HDFS log format.")

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

    template = message

    template = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", template)
    template = re.sub(r"\bblk_-?\d+\b", "<BLOCK>", template)
    template = re.sub(r":\d{4,5}\b", ":<PORT>", template)
    template = re.sub(r"\b\d+\b", "<NUM>", template)
    template = re.sub(r"\s+", " ", template).strip()

    template_freq = CACHED_TEMPLATE_FREQ.get(template, 0)
    component_freq = CACHED_COMPONENT_FREQ.get(component, 0)

    message_length = len(message)
    word_count = len(message.split())
    block_count = len(re.findall(r"\bblk_-?\d+\b", message))
    ip_count = len(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", message))

    exception_present = int("exception" in message.lower())
    verification_present = int("verification" in message.lower())
    delete_present = int(bool(re.search(r"delete|deleting", message, re.IGNORECASE)))
    allocate_present = int("allocateblock" in message.lower())

    hour = int(time[0:2]) if len(time) >= 2 and time[0:2].isdigit() else 0
    minute = int(time[2:4]) if len(time) >= 4 and time[2:4].isdigit() else 0

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
# Dashboard / Web Views
# ========================================
@app.route("/")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    global login_count
    error = None

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            error = "Invalid or missing CSRF token"
            return render_template("login.html", error=error), 400

        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            login_count += 1
            session.permanent = True
            session["logged_in"] = True
            return redirect("/")
        else:
            error = "Wrong username or password"

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/login-stats")
def login_stats():
    return jsonify({
        "total_logins": login_count
    })

# ========================================
# Diagnostic & Testing Routes
# ========================================
@app.route("/test")
def test():
    return "Server Working"

@app.route("/db-test")
def db_test():
    db = None
    try:
        db = get_db_conn()
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        return f"Database Connected: {result}"
    except Exception as e:
        return str(e), 500
    finally:
        release_db_conn(db)

@app.route("/dns-test")
def dns_test():
    import socket
    try:
        ip = socket.gethostbyname(DB_HOST)
        return f"DNS OK: {ip}"
    except Exception as e:
        return str(e), 500

# ========================================
# Prediction API
# ========================================
@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    db = None
    try:
        data = request.get_json()

        if not data or "log" not in data:
            return jsonify({
                "error": "Please provide a log field."
            }), 400

        log_line = data["log"]
        features, log_info = extract_features(log_line)

        prediction = model.predict(features)[0]
        anomaly_score = model.decision_function(features)[0]

        status = "ANOMALY" if prediction == -1 else "NORMAL"

        db = get_db_conn()
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

        return jsonify({
            "status": status,
            "anomaly_score": round(float(anomaly_score), 6),
            "log": log_info
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    finally:
        release_db_conn(db)

# ========================================
# Dashboard Stats & Logs API
# ========================================
@app.route("/stats")
def get_stats():
    db = None
    try:
        db = get_db_conn()
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
        return jsonify({
            "total_logs": total_logs,
            "anomalies": anomalies,
            "normal_logs": normal_logs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        release_db_conn(db)

@app.route("/logs", methods=["GET"])
def get_logs():
    db = None
    try:
        db = get_db_conn()
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
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        release_db_conn(db)

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
