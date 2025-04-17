import argparse
import io
import os
import json
from tempfile import TemporaryDirectory
import re

import h5py
import boto3
from types_boto3_s3.client import S3Client
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from tqdm import tqdm

from acronym_tools import create_gripper_marker

from semantic_grasping_datagen.annotation import Annotation, GraspLabel, PracticeResult, Judgement, JudgementLabel
from semantic_grasping_datagen.utils import list_s3_files

BUCKET_NAME = "prior-datasets"
DATA_PREFIX = "semantic-grasping/acronym/"
PRACTICE_PREFIX = "semantic-grasping/practice-results/"
JUDGEMENTS_PREFIX = "semantic-grasping/judgements/"

def get_args():
    parser = argparse.ArgumentParser(description="Explore and visualize annotations.")
    parser.add_argument("-v", "--visualize", nargs=3, metavar=("CATEGORY", "OBJ_ID", "GRASP_ID"), help="Visualize a specific observation.")
    parser.add_argument("-r", "--random-viz", action="store_true", help="Visualize a random observation.")
    parser.add_argument("-u", "--viz-uzer", help="Visualize all annotations from a specific user.")
    parser.add_argument("--user-hist", action="store_true")
    parser.add_argument("-p", "--practice-results", help="See practice results for a user.")
    return parser.parse_args()

def load_object_data(s3: S3Client, category: str, obj_id: str) -> tuple[trimesh.Scene, np.ndarray]:
    datafile_key = f"{DATA_PREFIX}grasps/{category}_{obj_id}.h5"
    with TemporaryDirectory() as tmpdir:
        datafile_path = os.path.join(tmpdir, "data.h5")
        s3.download_file(BUCKET_NAME, datafile_key, datafile_path)
        with h5py.File(datafile_path, "r") as f:
            mesh_fname: str = f["object/file"][()].decode("utf-8")
            mtl_fname = mesh_fname[:-len(".obj")] + ".mtl"
            mesh_path = os.path.join(tmpdir, os.path.basename(mesh_fname))
            mtl_path = os.path.join(tmpdir, os.path.basename(mtl_fname))
            mesh_pfx = DATA_PREFIX + os.path.dirname(mesh_fname) + "/"
            s3.download_file(BUCKET_NAME, f"{DATA_PREFIX}{mesh_fname}", mesh_path)
            s3.download_file(BUCKET_NAME, f"{DATA_PREFIX}{mtl_fname}", mtl_path)
            with open(mtl_path, "r") as mtl_f:
                for line in mtl_f.read().splitlines():
                    if m := re.fullmatch(r".+ (.+\.jpg)", line):
                        texture_fname = m.group(1)
                        assert texture_fname == os.path.basename(texture_fname), texture_fname
                        texture_path = os.path.join(tmpdir, texture_fname)
                        s3.download_file(BUCKET_NAME, f"{mesh_pfx}{texture_fname}", texture_path)

            T = np.array(f["grasps/transforms"])
            mesh_scale = f["object/scale"][()]
        obj_mesh = trimesh.load(mesh_path)
        obj_mesh = obj_mesh.apply_scale(mesh_scale)
        if isinstance(obj_mesh, trimesh.Scene):
            scene = obj_mesh
        elif isinstance(obj_mesh, trimesh.Trimesh):
            scene = trimesh.Scene([obj_mesh])
        else:
            raise ValueError("Unsupported geometry type")
    return scene, T

def visualize_judgement(s3: S3Client, judgement: Judgement):
    print(f"Judgement from user {judgement.user_id}")
    print(f"\tObject: {judgement.annotation.obj.object_category}_{judgement.annotation.obj.object_id}")
    print(f"\tGrasp ID: {judgement.annotation.grasp_id}")
    print(f"\tLabel: {judgement.judgement_label}")
    print(f"\tTime taken: {judgement.time_taken:.2f} sec")
    print(f"\tDescription: {judgement.annotation.grasp_description}")
    if judgement.judgement_label == JudgementLabel.INACCURATE:
        print(f"\tCorrected Description: {judgement.correct_grasp_description}")

    scene, T = load_object_data(s3, judgement.annotation.obj.object_category, judgement.annotation.obj.object_id)
    gripper_marker = create_gripper_marker(color=[0, 255, 0]).apply_transform(T[judgement.annotation.grasp_id])
    gripper_marker.apply_translation(-scene.centroid)
    scene.apply_translation(-scene.centroid)
    scene.add_geometry(gripper_marker)
    try:
        scene.to_mesh().show()
    except AttributeError:
        pass

def print_practice_results(s3: S3Client, user_id: str):
    practice_result_files = list_s3_files(s3, BUCKET_NAME, PRACTICE_PREFIX)
    for file in practice_result_files:
        if user_id in file:
            file_bytes = io.BytesIO()
            s3.download_fileobj(BUCKET_NAME, file, file_bytes)
            practice_result = PracticeResult.model_validate_json(file_bytes.getvalue())
            print(f"Study ID: {practice_result.study_id}")
            print(f"\tTimestamp: {practice_result.timestamp}")
            print(f"\tTotal time: {practice_result.total_time:.2f} sec")
            print(f"\tQuestion results:")
            for q in practice_result.question_results:
                print(f"\t\tQuestion {q.question_idx}: {q.correct} ({q.time_taken:.2f} sec)")

def parse_judgement_key(key: str):
    basename = os.path.basename(key)
    part1, annot_id = basename[:-len(".json")].split("___")
    study_id, user_id = part1.split("__")

    annot_id_parts = annot_id.split("__")
    if len(annot_id_parts) == 5:
        annot_id_parts = annot_id_parts[1:]
    object_category, object_id, grasp_id, annotator_id = annot_id_parts

    return {
        "key": key,
        "study_id": study_id,
        "user_id": user_id,
        "annot": {
            "object_category": object_category,
            "object_id": object_id,
            "grasp_id": int(grasp_id),
            "annotator_id": annotator_id
        }
    }

def get_judgement(s3: S3Client, local_dir: str, key: str):
    local_path = os.path.join(local_dir, os.path.basename(key))
    if not os.path.exists(local_path):
        print(f"Downloading {key} to {local_path}")
        s3.download_file(BUCKET_NAME, key, local_path)
    with open(local_path, "r") as f:
        return Judgement.model_validate_json(f.read())

def main():
    args = get_args()

    s3 = boto3.client("s3")

    judgement_keys = list_s3_files(s3, BUCKET_NAME, JUDGEMENTS_PREFIX)
    judgement_infos = [parse_judgement_key(key) for key in judgement_keys]

    local_dir = "data/judgements"

    print(f"Total judgements: {len(judgement_keys)}")

    if args.visualize:
        category, obj_id, grasp_id = args.visualize
        to_viz = []
        for info in judgement_infos:
            if info["annot"]["object_category"] == category and info["annot"]["object_id"] == obj_id and info["annot"]["grasp_id"] == int(grasp_id):
                to_viz.append(info)
        judgements = [get_judgement(s3, local_dir, info["key"]) for info in to_viz]
        print(f"Visualizing {len(judgements)} judgements")
        for judgement in judgements:
            visualize_judgement(s3, judgement)

    if args.random_viz:
        judgement_info = judgement_infos[np.random.randint(len(judgement_infos))]
        judgement = get_judgement(s3, local_dir, judgement_info["key"])
        visualize_judgement(s3, judgement)

    if args.viz_uzer:
        to_viz = []
        for info in judgement_infos:
            if info["user_id"] == args.viz_uzer:
                to_viz.append(info["key"])
        judgements = [get_judgement(s3, local_dir, key) for key in to_viz]
        print(f"User {args.viz_uzer} has submitted {len(judgements)} judgements.")
        for judgement in judgements:
            visualize_judgement(s3, judgement)

    if args.user_hist:
        user_hist = {}
        for info in judgement_infos:
            user_hist[info["user_id"]] = user_hist.get(info["user_id"], 0) + 1
        n_judgements = list(user_hist.values())
        plt.hist(n_judgements, bins=30)
        plt.xlabel("Number of judgements")
        plt.ylabel("Number of users")
        plt.title("Distribution of number of judgements per user")
        plt.axvline(x=np.mean(n_judgements), color="red", linestyle="--", label="Mean")
        plt.axvline(x=np.median(n_judgements), color="orange", linestyle="--", label="Median")
        plt.legend()
        plt.tight_layout(pad=0)
        plt.show()

    if args.practice_results:
        print_practice_results(s3, args.practice_results)

if __name__ == "__main__":
    main()
