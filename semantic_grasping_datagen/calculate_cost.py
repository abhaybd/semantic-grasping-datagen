import requests
import os

URL_BASE = "https://api.prolific.com/api/v1/"
PROJECT_ID = "67ae4305014ce47c5304d3d9"  # Project ID for Embodied AI
CURRENCY = "USD"

def get_studies(token: str, statuses: set[str] | None = None):
    if statuses is None:
        statuses = {"COMPLETED", "AWAITING_REVIEW"}

    response = requests.get(
        URL_BASE + f"projects/{PROJECT_ID}/studies",
        headers={"Authorization": f"Token {token}"}
    )
    response.raise_for_status()
    data = response.json()["results"]
    ret: list[tuple[str, str]] = []
    for study_info in data:
        if study_info["status"] in statuses and "semantic-grasping" in study_info["internal_name"]:
            ret.append((study_info["id"], study_info["internal_name"]))
    return ret

def get_study_cost(token: str, study_id: str):
    response = requests.get(
        URL_BASE + f"studies/{study_id}/cost",
        headers={"Authorization": f"Token {token}"}
    )
    response.raise_for_status()
    data = response.json()
    total = 0
    for category in ["rewards", "bonuses"]:
        for key in ["rewards", "fees", "tax"]:
            total += data[category][key]["amount"]
            assert CURRENCY == data[category][key]["currency"], f"{study_id} has currency {data[category][key]['currency']} not {CURRENCY}"
    return total / 100

def main():
    token = os.getenv("PROLIFIC_TOKEN")
    assert token is not None, "PROLIFIC_TOKEN is not set"

    studies = get_studies(token)
    studies.sort(key=lambda x: x[1])
    total_cost = 0
    max_name_len = max(len(name) for _, name in studies)
    for study_id, name in studies:
        total = get_study_cost(token, study_id)
        total_cost += total
        print(f"{name:{max_name_len}}: ${total:7.2f}")
    print(f"Total cost: ${total_cost:.2f}")

if __name__ == "__main__":
    main()
