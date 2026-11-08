import random
import time

import requests

URL = "http://127.0.0.1:8000"
random.seed(1)

for i in range(200):
    if i % 10 == 9:
        # deliberately wrong input so some requests fail with 422
        body = {"age": "abc", "monthly_income": 50000, "years_at_company": 3, "overtime": 1}
    else:
        body = {
            "age": random.randint(22, 60),
            "monthly_income": random.randint(20000, 150000),
            "years_at_company": random.randint(0, 20),
            "overtime": random.randint(0, 1),
        }
    r = requests.post(f"{URL}/predict", json=body)
    if i % 25 == 0:
        print(i, r.status_code, r.json())
    time.sleep(0.05)

print("health:", requests.get(f"{URL}/health").json())
