import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--scan-dir", type=str, default="/net/nfs2.prior/abhayd/semantic-grasping/data/taskgrasp/scans")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tg_library = TaskGraspScanLibrary(args.scan_dir)
    for elem in tqdm(tg_library):
        if elem["registered_grasps"] is None:
            continue

        registered_grasps = elem["registered_grasps"]
        rgb: Image.Image = elem["rgb"]
        cam_params = elem["cam_params"]

        grasp_pos = registered_grasps[:, :3, 3] + registered_grasps[:, :3, 2] * 0.09
        rgb_copy = rgb.copy()

        image_draw = ImageDraw.Draw(rgb_copy)
        for grasp_pos in grasp_pos:
            image_draw.ellipse((grasp_pos[0] - 10, grasp_pos[1] - 10, grasp_pos[0] + 10, grasp_pos[1] + 10), fill="red")

        rgb_copy.save(os.path.join(args.out_dir, f"{elem['object_id']}_{elem['scan_id']}.png"))
        break

if __name__ == "__main__":
    main()
