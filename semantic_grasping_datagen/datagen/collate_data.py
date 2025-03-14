import argparse
from collections import defaultdict
import os
import re
import yaml
import glob
import csv

from tqdm import tqdm
import torch
import numpy as np

from grasp_desc_encoder import GraspDescriptionEncoder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--full-precision", action="store_true")
    return parser.parse_args()

def embed_texts(batch_size: int, texts: list[str], full_precision: bool = False):
    encoder = GraspDescriptionEncoder("cuda", full_precision)
    text_embeddings = []
    for i in range(0, len(texts), batch_size):
        text_embeddings.append(encoder.encode(texts[i:i+batch_size]))
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    return text_embeddings

def main():
    args = get_args()

    texts = []
    dataset_path = os.path.join(args.out_dir, "dataset.csv")
    with open(dataset_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "text", "rgb_path", "xyz_path", "grasp_pose_path"])  # Write header

        for annotation_path in tqdm(
            glob.glob(os.path.join(args.observation_dir, "**/annot.yaml"), recursive=True),
            desc="Collating annotations"
        ):
            with open(annotation_path, "r") as f:
                annotation = yaml.safe_load(f)

            annot_id = annotation["annotation_id"]
            # annot_desc = annotation["annotation"]
            obj_name = " ".join(s.lower() for s in re.split(r"(?<!^)(?=[A-Z])", annot_id.split("_", 1)[0]))
            annot = f"The grasp is on the {obj_name}"
            grasp_path = os.path.relpath(annotation_path.replace("annot.yaml", "grasp_pose.npy"), args.observation_dir)
            assert os.path.isfile(os.path.join(args.observation_dir, grasp_path)), f"File {grasp_path} does not exist"

            par_dir = os.path.dirname(os.path.dirname(annotation_path))
            rgb_path = os.path.relpath(os.path.join(par_dir, "rgb.png"), args.observation_dir)
            xyz_path = os.path.relpath(os.path.join(par_dir, "xyz.npy"), args.observation_dir)
            assert os.path.isfile(os.path.join(args.observation_dir, rgb_path)), f"File {rgb_path} does not exist"
            assert os.path.isfile(os.path.join(args.observation_dir, xyz_path)), f"File {xyz_path} does not exist"

            writer.writerow([annot_id, annot, rgb_path, xyz_path, grasp_path])
            texts.append(annot)

    print("Embedding texts...")
    unique_text_idxs = defaultdict(list)
    for i, text in enumerate(texts):
        unique_text_idxs[text].append(i)

    # to save time, only embed unique texts and duplicate
    unique_texts = list(unique_text_idxs.keys())
    unique_texts_embeddings = embed_texts(args.batch_size, unique_texts, args.full_precision)
    text_embeddings = np.zeros((len(texts), unique_texts_embeddings.shape[1]), dtype=np.float32)
    for text, embedded_text in zip(unique_texts, unique_texts_embeddings):
        idxs = unique_text_idxs[text]
        text_embeddings[idxs] = embedded_text.astype(np.float32)
    assert np.all(np.isclose(np.linalg.norm(text_embeddings, axis=1), 1.0))

    np.save(os.path.join(args.out_dir, "text_embeddings.npy"), text_embeddings)

if __name__ == "__main__":
    main()
