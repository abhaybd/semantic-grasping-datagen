import argparse
import os

import pandas as pd
import numpy as np
import h5py
from PIL import Image
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("data_dir")
    parser.add_argument("csv_path")
    parser.add_argument("--format", type=str, default="robopoint", choices=["robopoint", "molmo"])
    return parser.parse_args()


def points_to_tuples(grasp_pts: np.ndarray):
    if grasp_pts.ndim == 1:
        grasp_pts = grasp_pts[None]
    assert grasp_pts.ndim == 2 and grasp_pts.shape[-1] == 2
    return f"[{', '.join([f'({x:.3f}, {y:.3f})' for x, y in grasp_pts])}]"

def create_robopoint_sample(scene_id: str, view_id: str, obs_id: str, image_path: str, grasp_desc: str, grasp_pt: np.ndarray):
    return {
        "id": f"{scene_id}-{view_id}-{obs_id}",
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nPoint to the grasp described by the following text:\n{grasp_desc}\n\nYour answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
            },
            {
                "from": "gpt",
                "value": points_to_tuples(grasp_pt)
            }
        ]
    }

def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def create_molmo_sample(scene_id: str, view_id: str, obs_id: str, image_path: str, grasp_desc: str, grasp_pt: np.ndarray):
    return {
        "id": f"{scene_id}-{view_id}-{obs_id}",
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"Point to the grasp described by the following text:\n{grasp_desc}"
            },
            {
                "from": "gpt",
                "value": point_to_xml(grasp_pt)
            }
        ]
    }

def get_grasp_point(grasp_pose: np.ndarray, cam_params: np.ndarray, width: int, height: int):
    # TODO: this should be done using closest-point queries to the mesh
    grasp_pos_m = grasp_pose[:3, 3] + grasp_pose[:3, 2] * 0.112
    points_px = grasp_pos_m @ cam_params.T
    points_px = points_px[:, :2] / points_px[:, 2:3]
    points_frac = points_px / np.array([width, height])
    return points_frac

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    sample_fn = create_robopoint_sample if args.format == "robopoint" else create_molmo_sample

    df = pd.read_csv(args.csv_path)
    lines = []
    for _, row in df.iterrows():
        scene_id = row["scene_id"]
        view_id = row["view_id"]
        obs_id = row["obs_id"]

        grasp_desc = row["annot"]

        img_relpath = os.path.join("images", f"{scene_id}-{view_id}.png")

        with h5py.File(os.path.join(args.data_dir, row["scene_path"]), "r") as f:
            if not os.path.exists(os.path.join(args.out_dir, img_relpath)):
                img: Image.Image = Image.fromarray(f[row["rgb_key"]][:]).convert("RGB")
                img.save(os.path.join(args.out_dir, img_relpath))

            img_h, img_w = f[row["rgb_key"]].shape[:-1]
            grasp_pose = f[row["grasp_pose_key"]][:]
            cam_params = f[view_id]["cam_params"][:]
            grasp_pt = get_grasp_point(grasp_pose, cam_params, img_w, img_h)

            sample = sample_fn(scene_id, view_id, obs_id, img_relpath, grasp_desc, grasp_pt)
            lines.append(sample)

    print(f"Generated {len(lines)} data points, saving to {os.path.join(args.out_dir, f'{args.format}_data.json')}")
    with open(os.path.join(args.out_dir, f"{args.format}_data.json"), "w") as f:
        json.dump(lines, f, indent=2)

if __name__ == "__main__":
    main()
