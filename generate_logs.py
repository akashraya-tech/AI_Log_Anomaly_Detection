import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "logdetect-api-key-2026")
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

for i in range(10):
    log = {
        "log": f"081111 23010{i%10} {100+i} INFO dfs.DataBlockScanner: Verification succeeded for blk_{987654321000000000+i}"
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=log,
            headers={"X-API-Key": API_KEY}
        )
        print(i + 1, response.status_code, response.json().get("status", "ERROR"))
    except requests.exceptions.ConnectionError:
        print(i + 1, "ERROR: Could not connect to", API_URL)