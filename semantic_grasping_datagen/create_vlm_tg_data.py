import argparse
import os
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--data-dir", type=str, default="/net/nfs2.prior/abhayd/semantic-grasping/data/taskgrasp")
    parser.add_argument("--format", type=str, default="robopoint", choices=["robopoint", "molmo"])
    parser.add_argument("--dedupe", action="store_true")
    return parser.parse_args()

def points_to_tuples(grasp_pts: np.ndarray):
    if grasp_pts.ndim == 1:
        grasp_pts = grasp_pts[None]
    assert grasp_pts.ndim == 2 and grasp_pts.shape[-1] == 2
    return f"[{', '.join([f'({x:.3f}, {y:.3f})' for x, y in grasp_pts])}]"

def create_robopoint_sample(completed: set[str] | None, object_id: str, scan_id: str, grasp_id: int, object_name: str, task: str, image_path: str, grasp_pt: np.ndarray):
    uid = f"{object_id}-{scan_id}-{task}"
    if completed is not None:
        if uid in completed:
            return None
        completed.add(uid)
    return {
        "id": f"{object_id}-{scan_id}-{grasp_id}-{task}",
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nPoint to where to grasp the {object_name} in order to {task}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
            },
            {
                "from": "gpt",
                "value": points_to_tuples(grasp_pt)
            }
        ]
    }

def point_to_xml(grasp_pt: np.ndarray, object_name: str, task: str):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = f"Where to grasp the {object_name} in order to {task}"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def create_molmo_sample(completed: set[str] | None, object_id: str, scan_id: str, grasp_id: int, object_name: str, task: str, image_path: str, grasp_pt: np.ndarray):
    uid = f"{object_id}-{scan_id}-{task}"
    if completed is not None:
        if uid in completed:
            return None
        completed.add(uid)
    return {
        "id": f"{object_id}-{scan_id}-{grasp_id}-{task}",
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"Point to where to grasp the {object_name} in order to {task}."
            },
            {
                "from": "gpt",
                "value": point_to_xml(grasp_pt, object_name, task)
            }
        ]
    }

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    object_grasp_tasks: dict[str, dict[int, set[str]]] = {}
    with open(os.path.join(args.data_dir, "task2_results.txt"), "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            part1, task_label = line.strip().split(":")
            object_id, grasp_id, task = part1.split("-")
            grasp_id = int(grasp_id)
            if task_label == "1":
                if object_id not in object_grasp_tasks:
                    object_grasp_tasks[object_id] = {}
                if grasp_id not in object_grasp_tasks[object_id]:
                    object_grasp_tasks[object_id][grasp_id] = set()
                object_grasp_tasks[object_id][grasp_id].add(task)

    create_sample_fn = create_robopoint_sample if args.format == "robopoint" else create_molmo_sample

    completed = set() if args.dedupe else None

    tg_library = TaskGraspScanLibrary(os.path.join(args.data_dir, "scans"))
    lines = []
    for elem in tqdm(tg_library):
        if elem["registered_grasps"] is None or elem["segmented_pc"] is None:
            continue

        object_id = elem["object_id"]
        object_name = elem["object_name"]
        scan_id = elem["scan_id"]
        registered_grasps = elem["registered_grasps"]
        segmented_pc = elem["segmented_pc"]
        rgb: Image.Image = elem["rgb"]
        cam_params: np.ndarray = elem["cam_params"]

        pc_kdtree = cKDTree(segmented_pc[:, :3])

        grasp_pos_m = registered_grasps[:, :3, 3] + registered_grasps[:, :3, 2] * 0.112
        closest_points_idxs = pc_kdtree.query(grasp_pos_m, k=1)[1]
        closest_points = segmented_pc[closest_points_idxs, :3]

        points_px = closest_points @ cam_params.T
        points_px = points_px[:, :2] / points_px[:, 2:3]
        points_frac = points_px / np.array([rgb.width, rgb.height])

        image_relpath = os.path.join("images", f"{object_id}_{scan_id}.png")
        rgb.save(os.path.join(args.out_dir, image_relpath))

        for grasp_id, grasp_pt in enumerate(points_frac):
            for task in object_grasp_tasks.get(object_id, {}).get(grasp_id, []):
                sample = create_sample_fn(completed, object_id, scan_id, grasp_id, object_name, task, image_relpath, grasp_pt)
                if sample is not None:
                    lines.append(sample)

    with open(os.path.join(args.out_dir, "taskgrasp_point.json"), "w") as f:
        json.dump(lines, f, indent=2)
    print(f"Generated {len(lines)} data points")

if __name__ == "__main__":
    main()
