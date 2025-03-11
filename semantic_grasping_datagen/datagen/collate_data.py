import argparse
from collections import defaultdict
import os
import yaml
import glob
import csv

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

QUERY_PFX = "Instruct: Given a description of a grasp, retrieve grasp descriptions that describe similar grasps on similar objects\nQuery: "

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def embed_texts(batch_size: int, texts: list[str]):
    nv_embed = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, model_kwargs={"torch_dtype": "bfloat16"})
    nv_embed.max_seq_length = 32768
    nv_embed.tokenizer.padding_side="right"
    nv_embed.eval()

    def add_eos(texts: list[str]):
        return [text + nv_embed.tokenizer.eos_token for text in texts]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeddings = nv_embed.encode(add_eos(texts), batch_size=batch_size, show_progress_bar=True, prompt=QUERY_PFX, normalize_embeddings=True)
    return text_embeddings

def main():
    args = get_args()

    texts = []
    dataset_path = os.path.join(args.out_dir, "dataset.csv")
    with open(dataset_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "text", "rgb_path", "xyz_path", "grasp_pose_path"])  # Write header

        for annotation_path in tqdm(
            glob.glob(os.path.join(args.observation_dir, "**/annot.yaml"), recursive=True),
            desc="Collating annotations"
        ):
            with open(annotation_path, "r") as f:
                annotation = yaml.safe_load(f)

            annot_id = annotation["annotation_id"]
            annot_desc = annotation["annotation"]
            grasp_path = os.path.relpath(annotation_path.replace("annot.yaml", "grasp_pose.npy"), args.observation_dir)
            assert os.path.isfile(os.path.join(args.observation_dir, grasp_path)), f"File {grasp_path} does not exist"

            par_dir = os.path.dirname(os.path.dirname(annotation_path))
            rgb_path = os.path.relpath(os.path.join(par_dir, "rgb.png"), args.observation_dir)
            xyz_path = os.path.relpath(os.path.join(par_dir, "xyz.npy"), args.observation_dir)
            assert os.path.isfile(os.path.join(args.observation_dir, rgb_path)), f"File {rgb_path} does not exist"
            assert os.path.isfile(os.path.join(args.observation_dir, xyz_path)), f"File {xyz_path} does not exist"

            writer.writerow([annot_id, annot_desc, rgb_path, xyz_path, grasp_path])
            texts.append(annot_desc)

    print("Embedding texts...")
    unique_text_idxs = defaultdict(list)
    for i, text in enumerate(texts):
        unique_text_idxs[text].append(i)

    # to save time, only embed unique texts and duplicate
    unique_texts = list(unique_text_idxs.keys())
    unique_texts_embeddings = embed_texts(args.batch_size, unique_texts)
    text_embeddings = np.zeros((len(texts), unique_texts_embeddings.shape[1]), dtype=np.float32)
    for text, embedded_text in zip(unique_texts, unique_texts_embeddings):
        idxs = unique_text_idxs[text]
        text_embeddings[idxs] = embedded_text.astype(np.float32)
    assert np.all(np.isclose(np.linalg.norm(text_embeddings, axis=1), 1.0))

    np.save(os.path.join(args.out_dir, "text_embeddings.npy"), text_embeddings)

if __name__ == "__main__":
    main()
