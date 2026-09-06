import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "pg-2cec95ed-ailoganomalydetection.a.aivencloud.com"),
    port=os.getenv("DB_PORT", "19268"),
    database=os.getenv("DB_NAME", "defaultdb"),
    user=os.getenv("DB_USER", "avnadmin"),
    password=os.getenv("DB_PASSWORD"),
    sslmode="require",
)

print("Connected Successfully!")

conn.close()