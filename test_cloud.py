import psycopg2

conn = psycopg2.connect(
    host="pg-2cec95ed-ailoganomalydetection.a.aivencloud.com",
    port="19268",
    database="defaultdb",
    user="avnadmin",
    password=os.getenv("DB_PASSWORD"),
    sslmode="require",
)

print("Connected Successfully!")

conn.close()