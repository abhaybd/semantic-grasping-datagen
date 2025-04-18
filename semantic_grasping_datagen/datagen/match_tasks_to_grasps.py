import argparse
import os
import json
import csv
from typing import Any, TypeAlias

import h5py
import numpy as np
import yaml
import pickle
from semantic_grasping_datagen.utils import tqdm
from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder

TasksSpec: TypeAlias = dict[str, dict[str, list[dict[str, Any]]]]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_json")
    parser.add_argument("obs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    return parser.parse_args()

def embed_texts(batch_size: int, texts: list[str], pbar_desc: str=None):
    encoder = GraspDescriptionEncoder("cuda")
    text_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=pbar_desc):
        text_embeddings.append(encoder.encode(texts[i:i+batch_size]))
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    return text_embeddings

def get_annotated_grasp_library(obs_dir: str, out_dir: str, embed_batch_size: int) -> dict[str, np.ndarray]:
    """
    Args:
        obs_dir: observations directory
        out_dir: output directory
    Returns:
        annotated_grasp_library: dictionary mapping annotated grasp descriptions to their embeddings
    """
    cache_path = os.path.join(out_dir, "annotated_grasp_library.pkl")
    if not os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            library = pickle.load(f)
        return library
    
    grasp_descriptions = set()
    for fn in os.listdir(obs_dir):
        if not fn.endswith(".hdf5"):
            continue

        with h5py.File(os.path.join(obs_dir, fn), "r") as f:
            for view_id in f.keys():
                for obs_id in f[view_id].keys():
                    if not obs_id.startswith("obs_"):
                        continue
                    annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                    grasp_descriptions.add(annotation["grasp_description"])

    grasp_descriptions = list(grasp_descriptions)
    grasp_embeddings = embed_texts(embed_batch_size, grasp_descriptions, "Embedding annotated grasp descriptions")

    library = {desc: embed for desc, embed in zip(grasp_descriptions, grasp_embeddings)}
    with open(cache_path, "wb") as f:
        pickle.dump(library, f)
    return library

def get_candidate_grasp_library(tasks_spec_path: str, out_dir: str, embed_batch_size: int) -> dict[str, np.ndarray]:
    cache_path = os.path.join(out_dir, "candidate_grasp_library.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(tasks_spec_path, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    candidate_grasps = []
    for category in tasks_spec.keys():
        candidate_grasps.extend(tasks_spec[category].keys())

    assert len(set(candidate_grasps)) == len(candidate_grasps), "Duplicate candidate grasps"

    embedded_candidate_grasps = embed_texts(embed_batch_size, candidate_grasps, "Embedding candidate grasp descriptions")
    ret = {desc: embed for desc, embed in zip(candidate_grasps, embedded_candidate_grasps)}
    with open(cache_path, "wb") as f:
        pickle.dump(ret, f)
    return ret

def main():
    args = get_args()

    annotated_grasp_library = get_annotated_grasp_library(args.obs_dir, args.out_dir, args.embed_batch_size)
    candidate_grasp_library = get_candidate_grasp_library(args.task_json, args.out_dir, args.embed_batch_size)

    with open(args.task_json, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    with open(os.path.join(args.out_dir, "matched_tasks.csv"), "w") as out_csv:
        writer = csv.DictWriter(out_csv, ["scene_path", "scene_id", "view_id", "obs_id", "task", "original_grasp_desc", "matching_grasp_desc"])
        writer.writeheader()

        for scene_file in tqdm(os.listdir(args.obs_dir)):
            if not scene_file.endswith(".hdf5"):
                continue

            scene_id = scene_file[:-len(".hdf5")]
            with h5py.File(os.path.join(args.obs_dir, scene_file), "r") as f:
                for view_id in f.keys():
                    embedded_annotated_grasps = []
                    obs_ids = []
                    annotated_grasp_descs = []
                    for obs_id in f[view_id].keys():
                        if not obs_id.startswith("obs_"):
                            continue
                        annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                        grasp_desc = annotation["grasp_description"]
                        embedded_annotated_grasps.append(annotated_grasp_library[grasp_desc])
                        obs_ids.append(obs_id)
                        annotated_grasp_descs.append(grasp_desc)
                    embedded_annotated_grasps = np.stack(embedded_annotated_grasps, axis=0)  # (n_obs, embed_dim)

                    object_names = list(f[view_id]["object_names"].asstr())
                    for name in object_names:
                        category = name.split("_", 1)[0]
                        if category not in tasks_spec:
                            continue

                        for original_grasp_desc, task_infos in tasks_spec[category].items():
                            original_grasp_embedding = candidate_grasp_library[original_grasp_desc]
                            similarities = embedded_annotated_grasps @ original_grasp_embedding
                            idx = np.argmax(similarities)
                            matching_obs_id = obs_ids[idx]
                            matching_grasp_desc = annotated_grasp_descs[idx]

                            for task_info in task_infos:
                                task = task_info["text"]
                                writer.writerow({
                                    "scene_path": scene_file,
                                    "scene_id": scene_id,
                                    "view_id": view_id,
                                    "obs_id": matching_obs_id,
                                    "task": task,
                                    "original_grasp_desc": original_grasp_desc,
                                    "matching_grasp_desc": matching_grasp_desc
                                })

if __name__ == "__main__":
    main()
