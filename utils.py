import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def register_pooling_hook(model):
    attn_store = {}

    def pool_hook(module, input, output):
        attn_store["weights"] = output[1]  # [B, 1, 1024]

    model.vision_encoder.head.attention.register_forward_hook(pool_hook)
    return attn_store

def generate_heatmap(pixel_values, attn_weights, img_size=512):
    """
    Returns a numpy image for Gradio
    """

    attn = attn_weights[0, 0]        # [N]
    attn = attn / (attn.max() + 1e-8)

    heatmap = attn.reshape(32, 32)

    heatmap_up = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    heatmap_np = heatmap_up.cpu().numpy()
    heatmap_np = (heatmap_np * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

    img = pixel_values[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 0.75, heatmap_color, 0.25, 0)

    return overlay 
