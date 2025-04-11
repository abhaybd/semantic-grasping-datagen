import argparse
import os
import random
import boto3
import base64
import urllib.parse
import json

from types_boto3_s3 import S3Client

from utils import list_s3_files

BUCKET_NAME = "prior-datasets"
JUDGEMENT_PREFIX = "semantic-grasping/judgements/"

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--url", default="http://localhost:3000/judgement")
    args.add_argument("--annot-prefix", default="semantic-grasping/annotations-synthetic/")
    args.add_argument("-p", "--prolific-code")
    args.add_argument("-r", "--prolific-rejection-code")
    args.add_argument("-s", "--prolific-study-id")
    args.add_argument("-o", "--output", help="File to write the URLs to. If not provided, the URLs will be printed to the console.")
    args.add_argument("--overwrite", action="store_true", help="Generate judgement URLs even for annotations that have already been judged.")
    args.add_argument("--schedule-length", type=int, default=5, help="The number of judgements per URL.")
    args.add_argument("--limit", type=int, help="The number of URLs to generate. The total number of judgements is limit * schedule_length.")
    return args.parse_args()

def parse_annot_id(annot_id: str):
    parts = annot_id.split("__")
    if len(parts) == 4:
        object_category, object_id, grasp_id, user_id = parts
        study_id = ""
    else:
        study_id, object_category, object_id, grasp_id, user_id = parts
    return {
        "object_category": object_category,
        "object_id": object_id,
        "grasp_id": grasp_id,
        "user_id": user_id,
        "study_id": study_id
    }

def get_annotation_ids(s3: S3Client, annot_prefix: str):
    ret: list[str] = []
    keys = list_s3_files(s3, BUCKET_NAME, annot_prefix)
    for key in keys:
        filename = os.path.basename(key)[:-len(".json")]
        ret.append(filename)
    return ret

def judged_annotation_ids(s3: S3Client):
    ret: list[str] = []
    keys = list_s3_files(s3, BUCKET_NAME, JUDGEMENT_PREFIX)
    for key in keys:
        filename = os.path.basename(key)[:-len(".json")]
        annot_id = filename.split("___")[1]
        ret.append(annot_id)
    return ret

def main():
    args = get_args()
    assert args.annot_prefix.endswith("/"), "annot_prefix must end with a slash"

    s3 = boto3.client("s3")

    annotation_ids = get_annotation_ids(s3, args.annot_prefix)
    if not args.overwrite:
        already_judged = set(judged_annotation_ids(s3))
        annotation_ids = [annot_id for annot_id in annotation_ids if annot_id not in already_judged]
    random.shuffle(annotation_ids)

    annotations = [parse_annot_id(annot_id) for annot_id in annotation_ids]

    limit = min(args.limit * args.schedule_length if args.limit else len(annotations), len(annotations))
    annotations = annotations[:limit]

    urls = []
    for i in range(0, len(annotations), args.schedule_length):
        schedule_items = annotations[i:i+args.schedule_length]
        schedule = {
            "idx": 0,
            "judgements": schedule_items
        }
        print(json.dumps(schedule, indent=2))
        schedule_encoded = urllib.parse.quote_plus(base64.b64encode(json.dumps(schedule).encode()).decode())
        url = f"{args.url}?judgement_schedule={schedule_encoded}"
        if args.prolific_code:
            url += f"&prolific_code={args.prolific_code}"
            if args.prolific_rejection_code:
                url += f"&prolific_rejection_code={args.prolific_rejection_code}"
        if args.prolific_study_id:
            url += f"&study_id={args.prolific_study_id}"
        urls.append(url)
    
    if args.output:
        with open(args.output, "w") as f:
            for url in urls:
                f.write(url + "\n")
    else:
        for url in urls:
            print(url)

if __name__ == "__main__":
    main()
