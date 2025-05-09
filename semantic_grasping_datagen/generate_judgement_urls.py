import argparse
import os
import random
import boto3
import base64
import urllib.parse
import json
from typing import Optional

from types_boto3_s3 import S3Client

from semantic_grasping_datagen.utils import list_s3_files

BUCKET_NAME = "prior-datasets"
SYNTHETIC_ANNOT_PREFIX = "semantic-grasping/annotations-synthetic/"
HUMAN_ANNOT_PREFIX = "semantic-grasping/annotations/"
JUDGEMENT_PREFIX = "semantic-grasping/judgements/"

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--url", default="http://localhost:3000/practice")
    args.add_argument("--synthetic", action="store_true", help="Use synthetic annotations.")
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
    """
    Returns list of <study_id>__<object_category>__<object_id>__<grasp_id>__<user_id> for all annotations.
    Might return some without study_id.
    """
    ret: list[str] = []
    keys = list_s3_files(s3, BUCKET_NAME, annot_prefix)
    for key in keys:
        filename = os.path.basename(key)[:-len(".json")]
        ret.append(filename)
    return ret

def judged_annotation_ids(s3: S3Client):
    """
    Returns list of <object_category>__<object_id>__<grasp_id>__<user_id> for all annotations that have been judged.
    """
    ret: list[str] = []
    keys = list_s3_files(s3, BUCKET_NAME, JUDGEMENT_PREFIX)
    for key in keys:
        filename = os.path.basename(key)[:-len(".json")]
        annot_id = filename.split("___")[1]
        ret.append(annot_id)
    return ret

def annot_id_to_judged_annot_id(annot_id: str):
    parts = annot_id.split("__")
    if len(parts) == 5:
        parts = parts[1:]
    return "__".join(parts)

def generate_urls(
    url_base: str,
    schedule_length: int,
    synthetic: bool=False,
    overwrite: bool=False,
    prolific_code: Optional[str] = None,
    prolific_rejection_code: Optional[str] = None,
    prolific_study_id: Optional[str] = None,
    limit: Optional[int] = None
) -> list[str]:
    s3 = boto3.client("s3")

    annot_prefix = SYNTHETIC_ANNOT_PREFIX if synthetic else HUMAN_ANNOT_PREFIX
    annotation_ids = get_annotation_ids(s3, annot_prefix)
    print(f"Total annotations to judge: {len(annotation_ids)}")
    if not overwrite:
        already_judged = set(judged_annotation_ids(s3))
        print(f"Already judged {len(already_judged)} annotations")
        annotation_ids = [annot_id for annot_id in annotation_ids if annot_id_to_judged_annot_id(annot_id) not in already_judged]
        print(f"{len(annotation_ids)} annotations left to judge")
    random.shuffle(annotation_ids)

    annotations = [parse_annot_id(annot_id) for annot_id in annotation_ids]

    limit = min(limit * schedule_length if limit else len(annotations), len(annotations))
    annotations = annotations[:limit]

    urls = []
    for i in range(0, len(annotations), schedule_length):
        schedule_items = annotations[i:i+schedule_length]
        schedule = {
            "idx": 0,
            "judgements": schedule_items
        }
        schedule_encoded = urllib.parse.quote_plus(base64.b64encode(json.dumps(schedule).encode()).decode())
        url = f"{url_base}?judgement_schedule={schedule_encoded}"
        if prolific_code:
            url += f"&prolific_code={prolific_code}"
            if prolific_rejection_code:
                url += f"&prolific_rejection_code={prolific_rejection_code}"
        if prolific_study_id:
            url += f"&study_id={prolific_study_id}"
        url += "&judgement=true"
        urls.append(url)
    return urls

def main():
    args = get_args()

    urls = generate_urls(
        url_base=args.url,
        schedule_length=args.schedule_length,
        synthetic=args.synthetic,
        overwrite=args.overwrite,
        prolific_code=args.prolific_code,
        prolific_rejection_code=args.prolific_rejection_code,
        prolific_study_id=args.prolific_study_id,
        limit=args.limit
    )

    if args.output:
        with open(args.output, "w") as f:
            for url in urls:
                f.write(url + "\n")
    else:
        for url in urls:
            print(url)

if __name__ == "__main__":
    main()
