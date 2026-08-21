import requests

for i in range(10):
    log = {
        "log": f"081111 23010{i%10} {100+i} INFO dfs.DataBlockScanner: Verification succeeded for blk_{987654321000000000+i}"
    }

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json=log
    )

    print(i + 1, response.status_code)