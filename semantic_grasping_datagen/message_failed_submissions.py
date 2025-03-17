import argparse
import io
import requests
import os

import boto3
from types_boto3_s3.client import S3Client

from annotation import PracticeResult
from utils import list_s3_files

BUCKET_NAME = "prior-datasets"
RESULTS_PFX = "semantic-grasping/practice-results/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=str, help="Study ID")
    return parser.parse_args()

def get_submission_code(token: str, study: str):
    response = requests.get(
        f"https://api.prolific.com/api/v1/studies/{study}/",
        headers={"Authorization": f"Token {token}"},
    )
    if response.ok:
        study_info = response.json()
        for code_info in study_info["completion_codes"]:
            if code_info["code_type"] == "COMPLETED":
                return code_info["code"]
        raise Exception("No completion code found")
    else:
        raise Exception(f"Failed to retrieve study info: {response.status_code} {response.text}")

def get_unapproved_participants(token: str, study: str):
    """Get user IDs of submissions that are awaiting review"""
    response = requests.get(
        f"https://api.prolific.com/api/v1/studies/{study}/submissions/",
        headers={"Authorization": f"Token {token}"}
    )
    if response.ok:
        submissions = response.json()
        ret = []
        for submission in submissions["results"]:
            if submission["status"] == "AWAITING REVIEW":
                ret.append(submission["participant_id"])
        return ret
    else:
        raise Exception(f"Failed to retrieve submissions: {response.status_code} {response.text}")

def did_participant_fail(s3: S3Client, submission_code: str, participant_id: str):
    result_bytes = io.BytesIO()
    files = list_s3_files(s3, BUCKET_NAME, f"{RESULTS_PFX}{submission_code}__{participant_id}")
    if len(files) > 1:
        print(f"WARN: Participant {participant_id} has multiple results files: {files}. Skipping.")
        return []
    s3.download_fileobj(BUCKET_NAME, files[0], result_bytes)
    result = PracticeResult.model_validate_json(result_bytes.getvalue())
    wrong_qs = [q.question_idx for q in result.question_results if not q.correct]
    return wrong_qs

def already_messaged(token: str,participant_id: str):
    response = requests.get(
        "https://api.prolific.com/api/v1/messages/",
        headers={"Authorization": f"Token {token}"},
        params={"user_id": participant_id}
    )
    if not response.ok:
        raise Exception(f"Failed to retrieve messages: {response.status_code} {response.text}")
    messages = response.json()
    return any("return" in m["body"] for m in messages["results"])

def send_message(token: str, study: str, participant_id: str, message: str):
    response = requests.post(
        "https://api.prolific.com/api/v1/messages/",
        headers={"Authorization": f"Token {token}"},
        json={
            "recipient_id": participant_id,
            "body": message,
            "study_id": study,
        }
    )
    if not response.ok:
        raise Exception(f"Failed to send message: {response.status_code} {response.text}")

def main():
    args = get_args()
    prolific_token = os.getenv("PROLIFIC_TOKEN")
    assert prolific_token is not None, "PROLIFIC_TOKEN is not set"
    s3 = boto3.client("s3")

    submission_code = get_submission_code(prolific_token, args.study)
    unapproved_participants = get_unapproved_participants(prolific_token, args.study)

    for participant in unapproved_participants:
        wrong_qs = did_participant_fail(s3, submission_code, participant)
        if len(wrong_qs) <= 1:
            print(f"WARN: Participant {participant} is unapproved but hasn't failed. Skipping.")
            continue

        if len(wrong_qs) == 3:
            q_text = "all of the"
        elif len(wrong_qs) == 2:
            ordinals = ["first", "second", "third"]
            q_text = "the " + " and ".join(ordinals[i] for i in wrong_qs)
        message = f"I see in my logs that you answered {q_text} practice questions incorrectly (note that the first answer is the one that's graded, even if you resubmit), which constitutes a failed comprehension check. Could you return your submission?"

        if not already_messaged(prolific_token, participant):
            print(f"Messaging {participant}, who failed {len(wrong_qs)} questions.")
            # TODO: use the request return endpoint instead of just sending a message
            send_message(prolific_token, args.study, participant, message)
        else:
            print(f"Already messaged {participant}, skipping.")

if __name__ == "__main__":
    main()
