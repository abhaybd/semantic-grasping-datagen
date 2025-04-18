import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import os

import boto3
from types_boto3_s3.client import S3Client

from semantic_grasping_datagen.utils import list_s3_files, tqdm
from semantic_grasping_datagen.annotation import Judgement, JudgementLabel


BUCKET_NAME = "prior-datasets"
JUDGEMENTS_PREFIX = "semantic-grasping/judgements/"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--n-workers", type=int, default=16)
    return parser.parse_args()

def filter_judgement(s3: S3Client, out_dir: str, judgement_key: str):
    judgement_bytes = BytesIO()
    s3.download_fileobj(BUCKET_NAME, judgement_key, judgement_bytes)
    judgement_bytes.seek(0)
    judgement = Judgement.model_validate_json(judgement_bytes.read())

    if judgement.judgement_label == JudgementLabel.UNCERTAIN:
        return 0

    annotation = judgement.annotation
    if judgement.judgement_label == JudgementLabel.INACCURATE:
        if judgement.correct_grasp_description is None:
            print(f"Judgement {judgement_key} is INACCURATE but correct_grasp_description is None!")
            return 0
        annotation = annotation.model_copy(update={"grasp_description": judgement.correct_grasp_description})

    annot_id = f"{annotation.study_id}__{annotation.obj.object_category}__{annotation.obj.object_id}__{annotation.grasp_id}__{annotation.user_id}"
    out_path = f"{out_dir}/{annot_id}.json"
    with open(out_path, "w") as f:
        f.write(annotation.model_dump_json())

    return 1


def main():
    args = get_args()
    s3 = boto3.client("s3")

    os.makedirs(args.out_dir, exist_ok=True)

    judgement_keys = list_s3_files(s3, BUCKET_NAME, JUDGEMENTS_PREFIX)
    print(f"Number of judgements to filter: {len(judgement_keys)}")

    n_filtered = 0
    with ThreadPoolExecutor(args.n_workers) as executor:
        futures = [executor.submit(filter_judgement, s3, args.out_dir, judgement_key) for judgement_key in judgement_keys]
        print(f"Filtering judgements...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            n_filtered += future.result()

    print(f"Number of annotations after filtering: {n_filtered}")

if __name__ == "__main__":
    main()
