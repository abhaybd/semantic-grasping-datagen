import io
import pickle
import boto3

s3 = boto3.client("s3")

BUCKET_NAME = "prior-datasets"
DATA_PREFIX = "semantic-grasping/acronym/"

with open("categories.txt", "r") as f:
    CATEGORIES = set(f.read().splitlines())

skeleton_bytes = io.BytesIO()
s3.download_fileobj(BUCKET_NAME, f"{DATA_PREFIX}annotation_skeleton.pkl", skeleton_bytes)
skeleton_bytes.seek(0)
skeleton: dict[str, dict[str, dict[int, bool]]] = pickle.load(skeleton_bytes)


n_annotations = 0
for c in CATEGORIES:
    for d in skeleton[c].values():
        n_annotations += sum(1 for _ in d.values())

print(n_annotations)
