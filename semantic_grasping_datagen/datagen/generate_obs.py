from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import multiprocessing as mp
from io import BytesIO
import os
import pickle
import random
from base64 import b64decode
from contextlib import contextmanager
import signal

import h5py
import hydra
from omegaconf import DictConfig
import yaml

if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
# pyrender spawns a lot of OMP threads, limiting to 1 significantly reduces overhead
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm
import pyrender
import pyrender.light
import trimesh
import torch
from pytorch3d.structures import Pointclouds

from semantic_grasping_datagen.annotation import Annotation

@contextmanager
def block_signals(signals: list[int]):
    previous_blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_blocked)

worker_id = mp.Value("i", 0)

def worker_init(img_size: tuple[int, int]):
    n_gpus = int(os.popen("nvidia-smi --list-gpus | wc -l").read())
    with worker_id.get_lock():
        gpu_id = worker_id.value % n_gpus
        worker_id.value += 1
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)

    height, width = img_size
    renderer = pyrender.OffscreenRenderer(width, height)
    globals()["renderer"] = renderer

def build_scene(data: dict[str, any]):
    glb_bytes = BytesIO(b64decode(data["glb"].encode("utf-8")))
    tr_scene: trimesh.Scene = trimesh.load(glb_bytes, file_type="glb")
    scene = pyrender.Scene.from_trimesh_scene(tr_scene)

    for light in data["lighting"]:
        light_type = getattr(pyrender.light, light["type"])
        light_args = light["args"]
        light_args["color"] = np.array(light_args["color"]) / 255.0
        light_node = pyrender.Node(light["args"]["name"], matrix=light["transform"], light=light_type(**light_args))
        scene.add_node(light_node)
    return scene

def set_camera(scene: pyrender.Scene, cam_K: np.ndarray, cam_pose: np.ndarray):
    cam = pyrender.camera.IntrinsicsCamera(
        fx=cam_K[0, 0],
        fy=cam_K[1, 1],
        cx=cam_K[0, 2],
        cy=cam_K[1, 2],
        name="camera",
    )
    cam_node = pyrender.Node(name="camera", camera=cam, matrix=cam_pose)
    for n in (scene.get_nodes(name=cam_node.name) or []):
        scene.remove_node(n)
    scene.add_node(cam_node)

    cam_light = pyrender.light.PointLight(intensity=2.0, name="camera_light")
    camera_light_node = pyrender.Node(name="camera_light", matrix=cam_pose, light=cam_light)
    for n in (scene.get_nodes(name=camera_light_node.name) or []):
        scene.remove_node(n)
    scene.add_node(camera_light_node)

def backproject(cam_K: np.ndarray, depth: np.ndarray):
    """
    Args:
        cam_K: camera intrinsic matrix (3, 3)
        depth: depth image (H, W)
    Returns:
        xyz: xyz coordinates of the points in the camera frame (H, W, 3)
    """
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    xyz = uvd @ np.expand_dims(np.linalg.inv(cam_K).T, axis=0)
    return xyz

def render(out_dir: str, scene_dir: str):
    scene_id = os.path.basename(scene_dir)
    out_scene_file = f"{out_dir}/{scene_id}.hdf5"
    if os.path.isfile(out_scene_file):
        print(f"Skipping {scene_id} because it already has observations")
        return 0

    with open(f"{scene_dir}/scene.pkl", "rb") as f:
        scene_data = pickle.load(f)
    all_annotations: dict[str, tuple[Annotation, np.ndarray]] = scene_data["annotations"]
    scene = build_scene(scene_data)

    renderer: pyrender.OffscreenRenderer = globals()["renderer"]
    renderer.viewport_height, renderer.viewport_width = scene_data["img_size"]

    view_observations: list[list[tuple[np.ndarray, Annotation, str]]] = []
    view_rgb: list[np.ndarray] = []
    view_xyz: list[np.ndarray] = []
    view_poses: list[np.ndarray] = []  # in standard camera axes conventions
    for view in scene_data["views"]:
        cam_K = np.array(view["cam_K"])
        cam_pose_trimesh = np.array(view["cam_pose"])
        set_camera(scene, cam_K, cam_pose_trimesh)

        standard_to_trimesh_cam_trf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        cam_pose_standard = cam_pose_trimesh @ standard_to_trimesh_cam_trf
        view_poses.append(cam_pose_standard)

        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        xyz = backproject(cam_K, depth).astype(np.float32)
        view_rgb.append(color)
        view_xyz.append(xyz)
        observations = []
        for annot_id in view["annotations_in_view"]:
            annot, grasp_pose = all_annotations[annot_id]
            grasp_pose_in_cam_frame = np.linalg.solve(cam_pose_standard, grasp_pose)
            observations.append((grasp_pose_in_cam_frame, annot, annot_id))
        view_observations.append(observations)

    view_xyz_torch = torch.from_numpy(np.stack(view_xyz, axis=0)).float().cuda()  # (B, H, W, 3)
    view_points_torch = view_xyz_torch.reshape(view_xyz_torch.shape[0], -1, 3)  # (B, H * W, 3)
    view_pc = Pointclouds(points=view_points_torch)
    view_normals = view_pc.estimate_normals()  # (B, H * W, 3)
    view_normals = view_normals.reshape_as(view_xyz_torch).cpu().numpy()  # (B, H, W, 3)

    n_observations = 0
    with block_signals([signal.SIGINT]):
        with h5py.File(out_scene_file, "w") as f:
            for view_idx, (rgb, xyz, normals, pose, observations) in enumerate(zip(view_rgb, view_xyz, view_normals, view_poses, view_observations)):
                view_group = f.create_group(f"view_{view_idx}")

                rgb_ds = view_group.create_dataset("rgb", data=rgb, compression="gzip")
                rgb_ds.attrs['CLASS'] = np.string_('IMAGE')
                rgb_ds.attrs['IMAGE_VERSION'] = np.string_('1.2')
                rgb_ds.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
                rgb_ds.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')

                view_group.create_dataset("xyz", data=xyz, compression="gzip")
                view_group.create_dataset("normals", data=normals, compression="gzip")
                view_group.create_dataset("view_pose", data=pose, compression="gzip")

                for obs_idx, (grasp_pose, annot, annot_id) in enumerate(observations):
                    obs_group = view_group.create_group(f"obs_{obs_idx}")
                    obs_group.create_dataset("grasp_pose", data=grasp_pose, compression="gzip")
                    annot_str = yaml.dump({
                        "annotation_id": annot_id,
                        "grasp_description": annot.grasp_description,
                        "object_description": annot.obj_description,
                        "object_category": annot.obj.object_category,
                        "object_id": annot.obj.object_id,
                        "grasp_id": annot.grasp_id
                    })
                    obs_group.create_dataset("annot", data=annot_str.encode("utf-8"))
                    n_observations += 1
    return n_observations

class DummyExecutor:
    def __init__(self, initializer, initargs, **kwargs):
        initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        ret = fn(*args, **kwargs)
        f = Future()
        f.set_result(ret)
        return f

@hydra.main(version_base=None, config_path="../../config", config_name="obs_gen.yaml")
def main(cfg: DictConfig):
    in_dir = cfg["scene_dir"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    nproc = cfg["n_proc"] or os.cpu_count()
    multiproc = nproc > 1
    generated_observations = 0
    with (ProcessPoolExecutor if multiproc else DummyExecutor)(
        max_workers=nproc,
        initializer=worker_init,
        initargs=(cfg["img_size"],)
    ) as executor:
        while True:
            scenes: set[str] = set(fn for fn in os.listdir(in_dir) if os.path.isdir(f"{in_dir}/{fn}"))
            processed_scenes: set[str] = set(fn.split(".")[0] for fn in os.listdir(out_dir) if fn.endswith(".hdf5"))
            print(f"Total processed scenes: {len(processed_scenes)}, generated observations: {generated_observations}")

            batch = list(scenes - processed_scenes)
            if len(batch) == 0:
                break
            random.shuffle(batch)  # shuffled to avoid different workers processing the same scenes
            batch = batch[:min(len(batch), 4 * nproc)]

            futures: list[Future] = []
            # if not multiproc, work happens here - otherwise happens in next loop
            for fn in tqdm(batch, desc="Rendering", dynamic_ncols=True, disable=multiproc):
                futures.append(executor.submit(render, out_dir, f"{in_dir}/{fn}"))
            for f in tqdm(as_completed(futures), total=len(futures), desc="Rendering", dynamic_ncols=True, smoothing=0, disable=(not multiproc)):
                generated_observations += f.result()


if __name__ == "__main__":
    main()
