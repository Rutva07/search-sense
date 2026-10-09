"""Exercise personalized completion and submit a completed search."""
import httpx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()
with httpx.Client(base_url=args.url, trust_env=False) as client:
    client.get("/health").raise_for_status()
    response = client.post("/complete", json={"user_id": "demo-user", "prefix": "best hotels in"})
    response.raise_for_status()
    print(response.json())
    response = client.post("/observe", json={"user_id": "demo-user", "query": "best hotels in paris"})
    response.raise_for_status()
    print(response.json())
