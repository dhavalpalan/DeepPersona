import requests

r = requests.post("http://localhost:5000/api/simulate", json={"force_bot": True})
print(r.json())