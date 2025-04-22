import argparse
import os
import json
import csv
from typing import Any, TypeAlias, Optional
from io import BytesIO

import h5py
import numpy as np
from pydantic import BaseModel
import yaml
import pickle

from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from semantic_grasping_datagen.utils import tqdm

TasksSpec: TypeAlias = dict[str, dict[str, list[dict[str, Any]]]]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_json")
    parser.add_argument("obs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retrieve", nargs="?", help="Retrieve a batch job with the given id")
    return parser.parse_args()

SYS_PROMPT = """
""".strip()


def get_annotated_grasp_library(obs_dir: str, out_dir: str) -> dict[str, dict[str, list[str]]]:
    """
    Args:
        obs_dir: observations directory
        out_dir: output directory
    Returns:
        annotated_grasp_library: dictionary mapping object categories to object ids to list of grasp descriptions
    """
    cache_path = os.path.join(out_dir, "annotated_grasp_library.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            library = pickle.load(f)
        return library
    
    grasp_dict: dict[str, dict[str, list[str]]] = {}  # category -> object_id -> list of grasp descriptions
    for fn in tqdm(os.listdir(obs_dir), desc="Loading annotated grasp descriptions"):
        if not fn.endswith(".hdf5"):
            continue

        with h5py.File(os.path.join(obs_dir, fn), "r") as f:
            for view_id in f.keys():
                for obs_id in f[view_id].keys():
                    if not obs_id.startswith("obs_"):
                        continue
                    annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                    obj_category = annotation["object_category"]
                    obj_id = annotation["object_id"]
                    grasp_desc = annotation["grasp_description"]
                    if obj_category not in grasp_dict:
                        grasp_dict[obj_category] = {}
                    if obj_id not in grasp_dict[obj_category]:
                        grasp_dict[obj_category][obj_id] = []
                    grasp_dict[obj_category][obj_id].append(grasp_desc)

    with open(cache_path, "wb") as f:
        pickle.dump(grasp_dict, f)

    return grasp_dict

def get_candidate_grasp_library(tasks_spec_path: str, out_dir: str) -> dict[str, list[str]]:
    cache_path = os.path.join(out_dir, "candidate_grasp_library.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(tasks_spec_path, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    grasp_dict: dict[str, list[str]] = {}
    for category in tasks_spec.keys():
        grasp_dict[category] = list(tasks_spec[category].keys())

    with open(cache_path, "wb") as f:
        pickle.dump(grasp_dict, f)

    return grasp_dict

class GraspMatch(BaseModel):
    candidate_grasp_desc: str
    matching_grasp_desc: Optional[str]

def create_query(object_category: str, object_id: str, candidate_grasp: str, annotated_grasps: list[str]):
    user_message = f"The object is a(n) {object_category}. The candidate grasp description is: \"{candidate_grasp}\". The annotated grasp descriptions are:"
    for grasp_desc in annotated_grasps:
        user_message += f"\n- {grasp_desc}"
    request = {
        "custom_id": f"{object_category}_{object_id}___{candidate_grasp}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grasp_match",
                    "strict": True,
                    "schema": to_strict_json_schema(GraspMatch)
                }
            }
        }
    }
    return request

def submit_matching_job(args):
    annotated_grasp_library = get_annotated_grasp_library(args.obs_dir, args.out_dir)
    candidate_grasp_library = get_candidate_grasp_library(args.task_json, args.out_dir)

    openai = OpenAI()
    queries = []
    for category, candidate_grasps in candidate_grasp_library.items():
        for object_id, annotated_grasps in annotated_grasp_library[category].items():
            for candidate_grasp in candidate_grasps:
                queries.append(create_query(category, object_id, candidate_grasp, annotated_grasps))

    batch_file = BytesIO()
    for query in queries:
        batch_file.write((json.dumps(query) + "\n").encode("utf-8"))
    print(f"Submitting {len(queries)} queries to OpenAI, total size: {batch_file.tell():,} bytes")
    # batch_file.seek(0)
    # batch_file_id = openai.files.create(file=batch_file, purpose="batch").id
    # batch = openai.batches.create(input_file_id=batch_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    # print(f"Submitted batch job with id: {batch.id}")
    # return batch.id

def retrieve_matching_job(openai: OpenAI, batch_id: str):
    batch = openai.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch job {batch_id} did not complete successfully!")
        return
    batch_file = openai.files.content(batch.output_file_id)
    

def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    annotated_grasp_library = get_annotated_grasp_library(args.obs_dir, args.out_dir)
    candidate_grasp_library = get_candidate_grasp_library(args.task_json, args.out_dir)

    if args.submit:
        batch_id = submit_matching_job(args)
    else:
        batch_id = args.batch_id

    retrieve_matching_job(args.openai, batch_id)

if __name__ == "__main__":
    main()
