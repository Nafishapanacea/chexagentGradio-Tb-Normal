import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

class TBModel(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values, output_attentions = True)
        embeddings = outputs.pooler_output
        return self.classifier(embeddings), outputs.attentions


def load_tb_model(checkpoint_path, device):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vision = AutoModel.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True
    ).vision_model

    model = TBModel(vision)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model
