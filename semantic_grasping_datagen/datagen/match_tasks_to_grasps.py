import argparse
import os
import json
import csv
from typing import Any, TypeAlias

import h5py
import numpy as np
import yaml
import pickle
import torch

from semantic_grasping_datagen.utils import tqdm
from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder

TasksSpec: TypeAlias = dict[str, dict[str, list[dict[str, Any]]]]

NVEMBED_QUERY_PFX = "Instruct: Given an instruction of how to grip an object, retrieve grasp descriptions that describe similar grasps on the same object.\nQuery: "

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_json")
    parser.add_argument("obs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    return parser.parse_args()

def embed_texts(encoder: GraspDescriptionEncoder, batch_size: int, texts: list[str], is_query: bool, pbar_desc: str=None):
    text_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=pbar_desc):
        text_embeddings.append(encoder.encode(texts[i:i+batch_size], is_query))
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    return text_embeddings

def get_annotated_grasp_library(obs_dir: str, out_dir: str, encoder: GraspDescriptionEncoder, embed_batch_size: int) -> dict[str, np.ndarray]:
    """
    Args:
        obs_dir: observations directory
        out_dir: output directory
    Returns:
        annotated_grasp_library: dictionary mapping annotated grasp descriptions to their embeddings
    """
    cache_path = os.path.join(out_dir, "annotated_grasp_library.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            library = pickle.load(f)
        return library
    
    grasp_descriptions = set()
    for fn in tqdm(os.listdir(obs_dir), desc="Loading annotated grasp descriptions"):
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
    grasp_embeddings = embed_texts(encoder, embed_batch_size, grasp_descriptions, False, "Embedding annotated grasp descriptions")

    library = {desc: embed for desc, embed in zip(grasp_descriptions, grasp_embeddings)}
    with open(cache_path, "wb") as f:
        pickle.dump(library, f)
    return library

def get_candidate_grasp_library(tasks_spec_path: str, out_dir: str, encoder: GraspDescriptionEncoder, embed_batch_size: int) -> dict[str, np.ndarray]:
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

    embedded_candidate_grasps = embed_texts(encoder, embed_batch_size, candidate_grasps, True, "Embedding candidate grasp descriptions")
    ret = {desc: embed for desc, embed in zip(candidate_grasps, embedded_candidate_grasps)}
    with open(cache_path, "wb") as f:
        pickle.dump(ret, f)
    return ret

def should_load_encoder(out_dir: str) -> bool:
    return not os.path.exists(os.path.join(out_dir, "annotated_grasp_library.pkl")) or not os.path.exists(os.path.join(out_dir, "candidate_grasp_library.pkl"))

def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    encoder = GraspDescriptionEncoder("cuda") if should_load_encoder(args.out_dir) else None
    annotated_grasp_library = get_annotated_grasp_library(args.obs_dir, args.out_dir, encoder, args.embed_batch_size)
    candidate_grasp_library = get_candidate_grasp_library(args.task_json, args.out_dir, encoder, args.embed_batch_size)
    del encoder
    torch.cuda.empty_cache()

    with open(args.task_json, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    n_samples = 0
    with open(os.path.join(args.out_dir, "matched_tasks.csv"), "w") as out_csv:
        writer = csv.DictWriter(out_csv, ["scene_path", "scene_id", "view_id", "obs_id", "task", "original_grasp_desc", "matching_grasp_desc"])
        writer.writeheader()

        for scene_file in tqdm(os.listdir(args.obs_dir), desc="Generating samples"):
            if not scene_file.endswith(".hdf5"):
                continue

            scene_id = scene_file[:-len(".hdf5")]
            with h5py.File(os.path.join(args.obs_dir, scene_file), "r") as f:
                for view_id in f.keys():
                    embedded_annotated_grasps: dict[str, tuple[list[np.ndarray], list[str], list[str]]] = {}  # category -> (list of embeddings, list of obs_ids, list of grasp_descs)
                    for obs_id in f[view_id].keys():
                        if not obs_id.startswith("obs_"):
                            continue
                        annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                        grasp_desc = annotation["grasp_description"]
                        category = annotation["object_category"]
                        if category not in embedded_annotated_grasps:
                            embedded_annotated_grasps[category] = ([], [], [])
                        embedded_annotated_grasps[category][0].append(annotated_grasp_library[grasp_desc])
                        embedded_annotated_grasps[category][1].append(obs_id)
                        embedded_annotated_grasps[category][2].append(grasp_desc)

                    object_names = list(f[view_id]["object_names"].asstr())
                    for name in object_names:
                        category = name.split("_", 1)[0]
                        if category not in tasks_spec or category not in embedded_annotated_grasps:
                            continue

                        for original_grasp_desc, task_infos in tasks_spec[category].items():
                            original_grasp_embedding = candidate_grasp_library[original_grasp_desc]

                            annotated_grasp_embeddings_for_obj, obs_ids_for_obj, grasp_descs_for_obj = embedded_annotated_grasps[category]
                            annotated_grasp_embeddings_for_obj = np.stack(annotated_grasp_embeddings_for_obj, axis=0)

                            similarities = annotated_grasp_embeddings_for_obj @ original_grasp_embedding
                            idx = np.argmax(similarities)
                            matching_obs_id = obs_ids_for_obj[idx]
                            matching_grasp_desc = grasp_descs_for_obj[idx]

                            # print("=" * 100)
                            # print(f"Original: {original_grasp_desc}\nMatching: {matching_grasp_desc}")

                            # ranking = np.argsort(similarities)[::-1]
                            # print("Rankings:")
                            # for i, idx in enumerate(ranking):
                            #     print(f"{i+1}. score={similarities[idx]:.3f}, {grasp_descs_for_obj[idx]}")

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
                                n_samples += 1

    print(f"Final dataset size: {n_samples:,}")

if __name__ == "__main__":
    main()
