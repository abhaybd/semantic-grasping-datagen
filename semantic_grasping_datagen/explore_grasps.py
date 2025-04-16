import os
import sys
from glob import glob
import yaml
import pickle
import random
import re

import h5py
from acronym_tools import create_gripper_marker
import trimesh
import numpy as np
import pyrender
import cv2

from semantic_grasping_datagen.synthetic_annotations import trimesh_to_pyrender


SCENE_DIR = "~/Desktop/grasp_scenes"
DUMP_DIR = "~/Desktop/grasp_dump"
DATA_DIR = "~/Desktop/grasping_data"


def create_scene(grasp: np.ndarray):
    scene = trimesh.Scene()

    standard_to_trimesh_cam_trf = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    grasp_trimesh = standard_to_trimesh_cam_trf @ grasp

    marker: trimesh.Trimesh = create_gripper_marker([0, 255, 0])
    marker.apply_transform(grasp_trimesh)
    scene.add_geometry(marker)

    return scene


def get_cam_k(scene_id, scene_dir=SCENE_DIR):
    scene_path = os.path.join(os.path.expanduser(scene_dir), scene_id, "scene.pkl")

    with open(scene_path, "rb") as f:
        scene_data = pickle.load(f)

    return [np.array(view["cam_K"]) for view in scene_data["views"]]


def render_gripper_rgbd(
    renderer: pyrender.OffscreenRenderer,
    grasp: np.ndarray,
    cam_k: np.ndarray,
):
    scene = trimesh_to_pyrender(create_scene(grasp))

    cam = pyrender.camera.IntrinsicsCamera(
        fx=cam_k[0, 0],
        fy=cam_k[1, 1],
        cx=cam_k[0, 2],
        cy=cam_k[1, 2],
        name="camera",
    )
    cam_node = pyrender.Node(name="camera", camera=cam, matrix=np.eye(4))
    scene.add_node(cam_node)

    cam_light = pyrender.PointLight(color=np.array([255, 255, 255]), intensity=0.5)
    cam_light_node = pyrender.Node(name="cam_light", light=cam_light, matrix=np.eye(4))
    scene.add_node(cam_light_node)

    view_arr, depth = renderer.render(scene, pyrender.RenderFlags.NONE)

    return view_arr, depth


def group_by_object_category(data_dir):
    if sys.platform == "darwin":
        print("removing", os.environ.pop("PYOPENGL_PLATFORM"), "from PYOPENGL_PLATFORM")

    renderer = pyrender.OffscreenRenderer(640, 480)

    dump_dir = os.path.expanduser(DUMP_DIR)
    os.makedirs(dump_dir, exist_ok=True)

    for file in glob(os.path.join(data_dir, "*.hdf5")):
        with h5py.File(file, "r") as f:
            scene_id = os.path.basename(file)[:-5]
            cam_ks = get_cam_k(scene_id)
            assert len(cam_ks) == len(f.keys())
            for view_id, cam_k in zip(f.keys(), cam_ks):
                rgb_key = f"{view_id}/rgb"
                xyz_key = f"{view_id}/xyz"
                normals_key = f"{view_id}/normals"
                for obs_id in f[view_id].keys():
                    if not obs_id.startswith("obs_"):
                        continue
                    if random.random() < 0.1:
                        annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                        print(
                            scene_id,
                            view_id,
                            obs_id,
                            annotation["object_category"],
                            annotation["grasp_description"],
                        )

                        grasp_pose_key = f"{view_id}/{obs_id}/grasp_pose"
                        annot_key = f"{view_id}/{obs_id}/annot"

                        grasp_pose = f[grasp_pose_key][()]

                        gripper, gripper_depth = render_gripper_rgbd(
                            renderer, grasp_pose, cam_k
                        )
                        gripper_mask = gripper_depth > 0
                        scene_depth = f[xyz_key][()][..., 2]
                        visible = gripper_mask * (scene_depth >= gripper_depth)
                        rgb = f[rgb_key][()]
                        rgb[visible] = gripper[visible]

                        from scipy.spatial.transform import Rotation as R

                        rot = R.from_matrix(grasp_pose[:3, :3]).as_euler(
                            "xyz", degrees=False
                        ) % (2 * np.pi)
                        if (
                            "vertical" in annotation["grasp_description"]
                            and not "horizontal" in annotation["grasp_description"]
                        ):
                            vertical_horizontal = "v"
                        elif (
                            "horizontal" in annotation["grasp_description"]
                            and not "vertical" in annotation["grasp_description"]
                        ):
                            vertical_horizontal = "h"
                        else:
                            vertical_horizontal = "unknown"

                        object_category = annotation["object_category"]
                        obj_name = " ".join(
                            s.lower()
                            for s in re.split(r"(?<!^)(?=[A-Z])", object_category)
                        )

                        for key in [
                            rgb_key,
                            xyz_key,
                            normals_key,
                            annot_key,
                            grasp_pose_key,
                        ]:
                            assert key in f, f"Key {key} not found in {scene_id}"

                        cv2.imshow("gripper", rgb)
                        cv2.waitKey(1)
                        fname = os.path.join(
                            dump_dir,
                            f"{os.path.basename(file)[:-5]}__{view_id}__{obs_id}__{obj_name}__{vertical_horizontal}.jpg",
                        )
                        cv2.imwrite(fname, rgb[:, :, ::-1])


if __name__ == "__main__":

    def main():
        group_by_object_category(os.path.expanduser(DATA_DIR))

    main()
    print("DONE")
