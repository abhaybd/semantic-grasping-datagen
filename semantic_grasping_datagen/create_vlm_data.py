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
    return parser.parse_args()

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
                lines.append({
                    "id": f"{object_id}-{scan_id}-{grasp_id}-{task}",
                    "image": image_relpath,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\nPoint to where to grasp the {object_name} in order to {task}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
                        },
                        {
                            "from": "gpt",
                            "value": f"[({grasp_pt[0]:.3f}, {grasp_pt[1]:.3f})]"
                        }
                    ]
                })

    with open(os.path.join(args.out_dir, "taskgrasp_point.json"), "w") as f:
        json.dump(lines, f, indent=2)
    print(f"Generated {len(lines)} data points")

if __name__ == "__main__":
    main()
