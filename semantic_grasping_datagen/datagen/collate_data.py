import argparse
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
        writer.writerow(["annotation_id", "text", "observation_path"])  # Write header
        
        for annotation_path in tqdm(glob.glob(os.path.join(args.observation_dir, "**/annot.yaml"), recursive=True)):
            with open(annotation_path, "r") as f:
                annotation = yaml.safe_load(f)

            annot_id = annotation["annotation_id"]
            annot_desc = annotation["annotation"]
            data_path = os.path.relpath(annotation_path.replace("annot.yaml", "obs.pkl"), args.observation_dir)
            assert os.path.isfile(os.path.join(args.observation_dir, data_path)), f"File {data_path} does not exist"

            writer.writerow([annot_id, annot_desc, data_path])
            texts.append(annot_desc)

    text_embeddings = embed_texts(args.batch_size, texts)
    np.save(os.path.join(args.out_dir, "text_embeddings.npy"), text_embeddings)

if __name__ == "__main__":
    main()
