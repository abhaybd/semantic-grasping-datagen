import argparse
import os
import glob
from typing import Any

from PIL import Image
import numpy as np
import trimesh
from tqdm import tqdm

from acronym_tools import create_gripper_marker

from semantic_grasping_datagen.eval.mask_detection import MaskDetector
from semantic_grasping_datagen.eval.pointcloud import CompositePCRegistration, DeepGMRRegistration, ICPRegistration

def img_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, mask: np.ndarray | None = None):
    h, w = rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask & mask]
    xyz = np.linalg.solve(cam_info, uvd.T).T
    return np.concatenate([xyz, rgb[depth_mask & mask]], axis=-1)

def pc_to_depth(xyz: np.ndarray, cam_info: np.ndarray, height: int, width: int):
    uvd = xyz @ cam_info.T
    uvd /= uvd[:, 2:]
    uv = uvd[:, :2].astype(np.int32)

    img = np.zeros((height, width), dtype=np.float32)
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[mask]
    xyz = xyz[mask]
    img[uv[:, 1], uv[:, 0]] = xyz[:, 2]
    return img

class TaskGraspScanLibrary:
    def __init__(self, tg_dir: str):
        assert os.path.isdir(tg_dir), "TaskGrasp directory does not exist"
        self.tg_dir = tg_dir
        self.rgb_paths = sorted(glob.glob(os.path.join(tg_dir, "*", "[0-9]_color.png")))

    def __len__(self):
        return len(self.rgb_paths)

    def get(self, object_id: str, scan_id: int):
        for i, rgb_path in enumerate(self.rgb_paths):
            dirname = os.path.dirname(rgb_path)
            if os.path.basename(dirname) == object_id:
                pass
            if os.path.basename(dirname) == object_id and os.path.basename(rgb_path).split("_", 1)[0] == str(scan_id):
                return self[i]
        raise ValueError(f"Scan {object_id}_{scan_id} not found")
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Returns (object_name, (rgb, depth, cam_params), fused_pc)"""
        dirname = os.path.dirname(self.rgb_paths[idx])
        object_id = os.path.basename(dirname)
        object_name = object_id.split("_", 1)[1].replace("_", " ")
        rgb_path = self.rgb_paths[idx]

        scan_id = os.path.basename(rgb_path).split("_", 1)[0]

        rgb = Image.open(rgb_path)
        rgb_array = np.asarray(rgb)
        depth = np.load(rgb_path[:-len("_color.png")] + "_depth.npy") / 1000.0
        cam_params = np.load(rgb_path[:-len("_color.png")] + "_camerainfo.npy")

        pc = img_to_pc(rgb_array, depth, cam_params)
        pc[:, 0] += 0.021
        pc[:, 1] -= 0.002
        corr_depth = pc_to_depth(pc[:,:3], cam_params, rgb_array.shape[0], rgb_array.shape[1])

        fused_pc = np.load(os.path.join(dirname, "fused_pc_clean.npy"))
        fused_pc[:,:3] -= np.mean(fused_pc[:,:3], axis=0)

        if os.path.exists(rgb_path[:-len("_color.png")] + "_registered_grasps.npy"):
            registered_grasps = np.load(rgb_path[:-len("_color.png")] + "_registered_grasps.npy")
        else:
            registered_grasps = None

        fused_grasps = []
        for grasp_dirname in sorted(os.listdir(os.path.join(dirname, "grasps")), key=int):
            grasp = np.load(os.path.join(dirname, "grasps", grasp_dirname, "grasp.npy"))
            fused_grasps.append(grasp)
        fused_grasps = np.array(fused_grasps)

        grasp_trf = np.array([
            [0, 0, 1, -0.09],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        fused_grasps = fused_grasps @ grasp_trf[None]

        return {
            "object_id": object_id,
            "scan_id": scan_id,
            "object_name": object_name,
            "rgb": rgb,
            "depth": corr_depth,
            "cam_params": cam_params,
            "fused_pc": fused_pc,
            "fused_grasps": fused_grasps,
            "registered_grasps": registered_grasps,
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=str, default="/net/nfs2.prior/abhayd/semantic-grasping/data/taskgrasp/scans")
    parser.add_argument("--gen-scene-dir", type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    scan_dir = args.scan_dir
    gen_scene_dir = args.gen_scene_dir
    if gen_scene_dir is not None:
        os.makedirs(gen_scene_dir, exist_ok=True)

    tg_library = TaskGraspScanLibrary(scan_dir)
    mask_detector = MaskDetector()
    pc_registration = CompositePCRegistration(
        DeepGMRRegistration(),
        ICPRegistration(),
    )

    for elem in tqdm(tg_library):
        out_grasps_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_registered_grasps.npy")
        out_pc_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_segmented_pc.npy")
        out_trf_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_pc_to_img_trf.npy")
        if not args.overwrite and os.path.isfile(out_grasps_file) and os.path.isfile(out_pc_file) and os.path.isfile(out_trf_file):
            continue
        object_mask = mask_detector.detect_mask(elem["object_name"], elem["rgb"])
        if object_mask is None:
            print(f"No mask detected for {elem['object_id']}_{elem['scan_id']}")
            continue
        rgb_array = np.asarray(elem["rgb"])
        object_pc = img_to_pc(rgb_array, elem["depth"], elem["cam_params"], object_mask)  # (N, 6)

        x_min, x_max = np.percentile(object_pc[:, 0], [2, 98])
        y_min, y_max = np.percentile(object_pc[:, 1], [2, 98])
        z_min, z_max = np.percentile(object_pc[:, 2], [2, 98])
        object_pc_crop = object_pc[
            (object_pc[:, 0] >= x_min) &
            (object_pc[:, 0] <= x_max) &
            (object_pc[:, 1] >= y_min) &
            (object_pc[:, 1] <= y_max) &
            (object_pc[:, 2] >= z_min) &
            (object_pc[:, 2] <= z_max)
        ]

        trf, object_pc_trf, cost = pc_registration.register(elem["fused_pc"][:,:3], object_pc_crop[:,:3])

        if cost >= 0.006:
            print(f"Failed to register {elem['object_id']}_{elem['scan_id']} with cost {cost}")
            continue

        trf_inv = np.linalg.inv(trf)
        fused_pc_registered = elem["fused_pc"][:, :3] @ trf_inv[:3,:3].T + trf_inv[:3,3]
        fused_pc_obj = trimesh.PointCloud(fused_pc_registered[:, :3], np.tile([255, 0, 0], (len(fused_pc_registered), 1)))

        scan_pc = img_to_pc(rgb_array, elem["depth"], elem["cam_params"], elem["depth"] < 2)
        scan_pc_obj = trimesh.PointCloud(scan_pc[:, :3], scan_pc[:, 3:].astype(np.uint8))

        scene = trimesh.Scene([fused_pc_obj, scan_pc_obj])
        grasps = trf_inv @ elem["fused_grasps"]
        for grasp in grasps:
            marker: trimesh.Trimesh = create_gripper_marker([255, 255, 0])
            marker.apply_transform(grasp)
            scene.add_geometry(marker)
        print(f"Registered {elem['object_id']}_{elem['scan_id']} with cost {cost}")
        np.save(out_grasps_file, grasps)
        np.save(out_pc_file, object_pc_crop)
        np.save(out_trf_file, trf_inv)
        if gen_scene_dir is not None:
            scene.export(os.path.join(gen_scene_dir, f"{elem['object_id']}_{elem['scan_id']}.glb"))

if __name__ == "__main__":
    main()
