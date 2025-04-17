import argparse
import requests
import os

import shortuuid

from semantic_grasping_datagen.generate_judgement_urls import generate_urls

URL_BASE = "https://api.prolific.com/api/v1/"
PROJECT_ID = "67ae4305014ce47c5304d3d9"  # Project ID for Embodied AI

SCHEDULE_LENGTH = 5
DURATION = 5  # minutes
HOURLY_RATE = 1200  # cents

DISTRIBUTION_RATE = 0.5

STUDY_DESCRIPTION = f"""
<h2>Overview</h2>

<p>
In this study, we are seeking to annotate data which will be used to help robots understand how to grasp and manipulate different types of everyday objects. Using a keyboard and mouse, participants will use a web interface to filter descriptions of 3D objects and grasps.
</p>

<p>
This is a small test-run to iron out some issues. If you encounter any issues or have any feedback, please message the researchers!
</p>

<h2>Study Details</h2>

<p>
You will be shown a 3D view of an object and a grasp on the object, along with a description of how that object is being grasped. Your task is to determine if that description is accurate or not. Further instructions and details are available in the study. The study consists of {SCHEDULE_LENGTH} annotations, and should take no more than {DURATION} minutes. All answers should be in English.
</p>

<h2>About Us</h2>

<p>
We are a team of researchers at the Allen Institute for Artificial Intelligence (Ai2), a non-profit committed to truly open and world-class AI research. For more information about Ai2 visit <a href="https://allenai.org">https://allenai.org</a>.
</p>
""".strip()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("internal_name")
    return parser.parse_args()

def create_study(token: str, internal_name: str, study_urls: list[str], accept_code: str, reject_code: str):
    access_details = [{"external_url": url, "total_allocation": 1} for url in study_urls]
    response = requests.post(
        URL_BASE + f"studies",
        headers={"Authorization": f"Token {token}"},
        json={
            "name": "Robotics Grasp Description Filtering",
            "internal_name": internal_name,
            "description": STUDY_DESCRIPTION,
            "access_details": access_details,
            "prolific_id_option": "question",
            "completion_codes": [
                {
                    "code": accept_code,
                    "code_type": "COMPLETED",
                    "actions": [{"action": "AUTOMATICALLY_APPROVE"}]
                },
                {
                    "code": reject_code,
                    "code_type": "FAILED_ATTENTION_CHECK",
                    "actions": [
                        {
                            "action": "REQUEST_RETURN",
                            "return_reason": "Answered too many practice questions incorrectly, constituting a failed comprehension check."
                        }
                    ]
                }
            ],
            "total_available_places": len(study_urls),
            "estimated_completion_time": DURATION,
            "reward": int(DURATION * HOURLY_RATE / 60),
            "device_compatibility": ["desktop"],
            "filters": [
                {
                    "filter_id": "ai-taskers",
                    "selected_values": ["0"]
                },
                {
                    "filter_id": "fluent-languages",
                    "selected_values": ["19"],
                },
                {
                    "filter_id": "current-country-of-residence",
                    "selected_values": ["0", "1"],
                },
            ],
            "naivety_distribution_rate": DISTRIBUTION_RATE,
            "project": PROJECT_ID,
            "submissions_config": {
                "max_submissions_per_participant": 100,
                "max_concurrent_submissions": -1
            },
            "study_labels": ["annotation"]
        }
    )

    response.raise_for_status()
    data = response.json()
    if response.status_code == 201:
        print("Successfully created study!")
        print("Study ID:", data["id"])
    else:
        raise ValueError(f"Failed to create study! Error={data}")

def main():
    args = get_args()

    token = os.getenv("PROLIFIC_TOKEN")
    assert token is not None, "PROLIFIC_TOKEN is not set"

    prolific_code = shortuuid.uuid()
    prolific_rejection_code = shortuuid.uuid()

    urls = generate_urls(
        url_base="http://semantic-grasping-annotation-alb-378791810.us-east-2.elb.amazonaws.com/practice",
        schedule_length=SCHEDULE_LENGTH,
        synthetic=True,
        overwrite=False,
        prolific_code=prolific_code,
        prolific_rejection_code=prolific_rejection_code,
        prolific_study_id="{{%STUDY_ID%}}"
    )
    print(urls[:2])

    create_study(token, args.internal_name, urls, prolific_code, prolific_rejection_code)

if __name__ == "__main__":
    main()
