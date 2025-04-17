import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
import threading

import pandas as pd
import numpy as np
import h5py
from PIL import Image
import json

from semantic_grasping_datagen.utils import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("data_dir")
    parser.add_argument("csv_path")
    parser.add_argument("--format", type=str, default="robopoint", choices=["robopoint", "molmo"])
    parser.add_argument("--n-proc", type=int, default=16)
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

def copy_image(scene_path: str, scene_id: str, view_id: str, rgb_key: str, data_dir: str, out_dir: str):
    img_relpath = os.path.join("images", f"{scene_id}-{view_id}.png")

    with h5py.File(os.path.join(data_dir, scene_path), "r") as f:
        img: Image.Image = Image.fromarray(f[rgb_key][:]).convert("RGB")
    img.save(os.path.join(out_dir, img_relpath))

def copy_images(df: pd.DataFrame, data_dir: str, out_dir: str, n_proc: int):
    unique_views = set()
    for _, row in df.iterrows():
        scene_path = row["scene_path"]
        scene_id = row["scene_id"]
        view_id = row["view_id"]
        rgb_key = row["rgb_key"]
        unique_views.add((scene_path, scene_id, view_id, rgb_key))

    with ThreadPoolExecutor(max_workers=n_proc) as executor:
        futures = []
        for scene_path, scene_id, view_id, rgb_key in unique_views:
            futures.append(executor.submit(copy_image, scene_path, scene_id, view_id, rgb_key, data_dir, out_dir))
        wait(futures)

def create_sample(data_dir: str, row: pd.Series, format: str):
    sample_fn = create_robopoint_sample if format == "robopoint" else create_molmo_sample
    scene_id = row["scene_id"]
    view_id = row["view_id"]
    obs_id = row["obs_id"]

    grasp_desc = row["annot"]

    img_relpath = os.path.join("images", f"{scene_id}-{view_id}.png")

    with h5py.File(os.path.join(data_dir, row["scene_path"]), "r") as f:
        grasp_pt = f[row["view_id"]][row["obs_id"]]["grasp_point_px"][:]

    sample = sample_fn(scene_id, view_id, obs_id, img_relpath, grasp_desc, grasp_pt)
    return sample


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    df = pd.read_csv(args.csv_path)

    copy_thread = threading.Thread(target=copy_images, args=(df, args.data_dir, args.out_dir, args.n_proc))
    copy_thread.start()

    lines: list[str] = []
    submit_semaphore = threading.Semaphore(4 * args.n_proc)
    with ProcessPoolExecutor(max_workers=args.n_proc) as executor:
        with tqdm(total=len(df), desc="Constructing samples") as pbar:
            def on_job_done(_):
                submit_semaphore.release()
                pbar.update(1)

            futures: list[Future] = []
            for _, row in df.iterrows():
                submit_semaphore.acquire()
                future = executor.submit(create_sample, args.data_dir, row, args.format)
                future.add_done_callback(on_job_done)
                futures.append(future)
            for future in as_completed(futures):
                lines.append(future.result())

    print(f"Generated {len(lines)} data points, saving to {os.path.join(args.out_dir, f'{args.format}_data.json')}")
    with open(os.path.join(args.out_dir, f"{args.format}_data.json"), "w") as f:
        json.dump(lines, f, indent=2)

    if copy_thread.is_alive():
        print("Waiting for image copy thread to finish...")
    copy_thread.join()

if __name__ == "__main__":
    main()
