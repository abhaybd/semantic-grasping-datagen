import numpy as np
import torch
from torch import nn

from nv_embed_v2 import NVEmbedModel

class GraspDescriptionEncoder(nn.Module):
    QUERY_PFX = "Instruct: Given a description of a grasp, retrieve grasp descriptions that describe similar grasps on similar objects\nQuery: "
    MAX_LENGTH = 32768

    def __init__(self, device: str = "cpu", full_precision: bool = False):
        super().__init__()
        kwargs = {"torch_dtype": "bfloat16"} if full_precision else {}
        self.nv_embed = NVEmbedModel.from_pretrained("nvidia/NV-Embed-v2", map_location=device, **kwargs)
        self.nv_embed.eval()

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, descriptions: list[str]):
        return self.nv_embed.encode(descriptions, instruction=self.QUERY_PFX, max_length=self.MAX_LENGTH)

    def encode(self, descriptions: list[str]) -> np.ndarray:
        return self(descriptions).cpu().numpy()
