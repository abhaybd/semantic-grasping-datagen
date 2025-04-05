import argparse
from collections import defaultdict
import os
import re
import csv

import yaml
from tqdm import tqdm
import h5py
import numpy as np

from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--annot-type", type=str, default="full", choices=["objectnames", "grasp_description", "full"])
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

    os.makedirs(args.out_dir, exist_ok=True)

    texts = []
    dataset_path = os.path.join(args.out_dir, "dataset.csv")
    with open(dataset_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene_path", "annotation_id", "scene_id", "view_id", "rgb_key", "xyz_key", "normals_key", "annot_key", "grasp_pose_key", "annot"])

        for scene_file in tqdm(os.listdir(args.observation_dir), desc="Processing scenes"):
            if not scene_file.endswith(".hdf5"):
                continue

            scene_id = scene_file.split(".")[0]
            with h5py.File(os.path.join(args.observation_dir, scene_file), "r") as f:
                for view_id in f.keys():
                    rgb_key = f"{view_id}/rgb"
                    xyz_key = f"{view_id}/xyz"
                    normals_key = f"{view_id}/normals"
                    for obs_id in f[view_id].keys():
                        if not obs_id.startswith("obs_"):
                            continue
                        annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                        grasp_pose_key = f"{view_id}/{obs_id}/grasp_pose"
                        annot_key = f"{view_id}/{obs_id}/annot"

                        annot_id = annotation["annotation_id"]
                        object_category = annotation["object_category"]
                        obj_name = " ".join(s.lower() for s in re.split(r"(?<!^)(?=[A-Z])", object_category))
                        if args.annot_type == "objectnames":
                            annot = f"The grasp is on the {obj_name}."
                        elif args.annot_type == "grasp_description":
                            annot = annotation["grasp_description"]
                        elif args.annot_type == "full":
                            annot = f"The grasp is on the {obj_name}. " + annotation["grasp_description"]
                        texts.append(annot)

                        for key in [rgb_key, xyz_key, normals_key, annot_key, grasp_pose_key]:
                            assert key in f, f"Key {key} not found in {scene_file}"
                        writer.writerow([scene_file, annot_id, scene_id, view_id, rgb_key, xyz_key, normals_key, annot_key, grasp_pose_key, annot])

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
