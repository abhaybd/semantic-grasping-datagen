import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n-workers", type=int, default=16)
    return parser.parse_args()

def repackage_scene(data_dir: str, scene_id: str, output_dir: str):
    scene_dir = os.path.join(data_dir, scene_id)
    output_file = os.path.join(output_dir, f"{scene_id}.hdf5")
    
    with h5py.File(output_file, 'w') as f:
        for view_name in os.listdir(scene_dir):
            view_dir = os.path.join(scene_dir, view_name)
            if not os.path.isdir(view_dir):
                continue

            rgb_path = os.path.join(view_dir, 'rgb.png')
            rgb = np.asarray(Image.open(rgb_path))
            ds = f.create_dataset(f"{view_name}/rgb", data=rgb, compression="gzip")
            ds.attrs['CLASS'] = np.string_('IMAGE')
            ds.attrs['IMAGE_VERSION'] = np.string_('1.2')
            ds.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
            ds.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')

            xyz_path = os.path.join(view_dir, 'xyz.npy')
            xyz_data = np.load(xyz_path).astype(np.float32)
            f.create_dataset(f"{view_name}/xyz", data=xyz_data, compression="gzip")

            for obs_name in os.listdir(view_dir):
                obs_dir = os.path.join(view_dir, obs_name)
                if not os.path.isdir(obs_dir) or not obs_name.startswith('obs_'):
                    continue

                annot_path = os.path.join(obs_dir, 'annot.yaml')
                with open(annot_path, 'r') as yaml_file:
                    f.create_dataset(f"{view_name}/{obs_name}/annot", data=yaml_file.read())

                grasp_path = os.path.join(obs_dir, 'grasp_pose.npy')
                grasp_data = np.load(grasp_path)
                f.create_dataset(f"{view_name}/{obs_name}/grasp_pose", data=grasp_data)

def main():
    args = get_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    scene_ids = os.listdir(data_dir)
    nproc = args.n_workers
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        with tqdm(total=len(scene_ids), desc="Repackaging scenes") as pbar:
            futures = []
            n_submitted = 0
            while n_submitted < len(scene_ids):
                not_done = [f for f in futures if not f.done()]
                if len(not_done) < 4 * nproc:
                    futures.append(executor.submit(repackage_scene, data_dir, scene_ids[n_submitted], output_dir))
                    n_submitted += 1
                else:
                    time.sleep(0.5)
                
                new_futures = []
                for f in futures:
                    if f.done():
                        f.result()
                        pbar.update(1)
                    else:
                        new_futures.append(f)
                futures = new_futures
            if len(futures) > 0:
                for f in as_completed(futures):
                    f.result()
                    pbar.update(1)

if __name__ == "__main__":
    main()
