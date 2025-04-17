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

def download_files(s3: S3Client, local_dir: str, prefix: str):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    remote_files = list_s3_files(s3, BUCKET_NAME, prefix)
    files_to_download = []
    for key in remote_files:
        local_path = os.path.join(local_dir, os.path.basename(key))
        if not os.path.exists(local_path):
            files_to_download.append((key, local_path))

    for key, local_path in tqdm(files_to_download, desc="Downloading files", disable=len(files_to_download) == 0):
        s3.download_file(BUCKET_NAME, key, local_path)

def process_judgements(local_dir: str):
    judgements: list[Judgement] = []
    for filename in os.listdir(local_dir):
        if filename.endswith(".json"):
            with open(os.path.join(local_dir, filename), "r") as f:
                data = Judgement.model_validate_json(f.read())
                judgements.append(data)
    return judgements

def plot_object_distribution(annotations: list[Annotation]):
    object_counts = {}
    for annotation in annotations:
        object_counts[annotation.obj.object_category] = object_counts.get(annotation.obj.object_category, 0) + 1
    keys = sorted(object_counts.keys())
    plt.bar(keys, [object_counts[k] for k in keys])
    plt.xlabel("Object Category")
    plt.ylabel("Count")
    plt.title("Object Distribution")
    plt.show()


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

def visualize_judgement(judgement: Judgement):
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

def visualize_annotation(annotation: Annotation):
    print(f"Annotation from user {annotation.user_id}")
    print(f"\tObject: {annotation.obj.object_category}_{annotation.obj.object_id}")
    print(f"\tGrasp ID: {annotation.grasp_id}")
    print(f"\tLabel: {annotation.grasp_label}")
    print(f"\tMesh Malformed: {annotation.is_mesh_malformed}")
    print(f"\tTime taken: {annotation.time_taken:.2f} sec")
    print(f"\tObject Description: {annotation.obj_description}")
    print(f"\tGrasp Description: {annotation.grasp_description}")

    scene, T = load_object_data(s3, annotation.obj.object_category, annotation.obj.object_id)
    gripper_marker = create_gripper_marker(color=[0, 255, 0]).apply_transform(T[annotation.grasp_id])
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

def plot_judgement_distribution(judgements: list[Judgement]):
    judgement_counts = {}
    for judgement in judgements:
        label = judgement.judgement_label.name
        judgement_counts[label] = judgement_counts.get(label, 0) + 1
    keys = sorted(judgement_counts.keys())
    counts = [judgement_counts[k] for k in keys]
    total = sum(counts)
    percentages = [count/total * 100 for count in counts]
    
    fig, ax = plt.subplots()
    
    bars = ax.bar(keys, counts)
    ax.set_xlabel("Judgement Label")
    ax.set_ylabel("Count")
    
    # Add percentage labels on top of each bar
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore and visualize annotations.")
    parser.add_argument("--plot-object-dist", action="store_true", help="Plot histogram of object distribution.")
    parser.add_argument("--plot-judgement-dist", action="store_true", help="Plot judgement distribution.")
    parser.add_argument("-v", "--visualize", nargs=3, metavar=("CATEGORY", "OBJ_ID", "GRASP_ID"), help="Visualize a specific observation.")
    parser.add_argument("-r", "--random-viz", action="store_true", help="Visualize a random observation.")
    parser.add_argument("-u", "--viz-uzer", help="Visualize all annotations from a specific user.")
    parser.add_argument("--filtered", action="store_true", help="Use filtered annotations.")
    parser.add_argument("--user-hist", action="store_true")
    parser.add_argument("-p", "--practice-results", help="See practice results for a user.")
    args = parser.parse_args()

    s3 = boto3.client("s3")

    local_dir = "data/judgements"
    download_files(s3, local_dir, JUDGEMENTS_PREFIX)
    judgements = process_judgements(local_dir)

    print(f"Loaded {len(judgements)} judgements.")

    if args.plot_object_dist:
        plot_object_distribution([j.annotation for j in judgements])

    if args.plot_judgement_dist:
        plot_judgement_distribution(judgements)

    if args.visualize:
        category, obj_id, grasp_id = args.visualize
        judgement = next((j for j in judgements if j.annotation.obj.object_category == category and j.annotation.obj.object_id == obj_id and j.annotation.grasp_id == int(grasp_id)), None)
        visualize_judgement(judgement)

    if args.random_viz:
        judgement: Judgement = np.random.choice(judgements)
        visualize_judgement(judgement)

    if args.viz_uzer:
        for judgement in judgements:
            if judgement.user_id == args.viz_uzer:
                visualize_judgement(judgement)

    if args.user_hist:
        user_hist = {}
        for judgement in judgements:
            user_hist[judgement.user_id] = user_hist.get(judgement.user_id, 0) + 1
        users = [user for user in user_hist.keys() if user_hist[user] > 1]
        plt.bar(users, [user_hist[user] for user in users])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=0)
        plt.show()

    if args.practice_results:
        print_practice_results(s3, args.practice_results)
