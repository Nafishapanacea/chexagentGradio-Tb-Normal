import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

class CheXagentSigLIPBinary(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.view_embedding = nn.Embedding(3, 8)
        self.sex_embedding  = nn.Embedding(2, 4)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim + 8 + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, pixel_values, view, sex):
        outputs = self.vision_encoder(pixel_values=pixel_values, output_attentions = True)
        embeddings = outputs.pooler_output

        view_emb = self.view_embedding(view)
        sex_emb  = self.sex_embedding(sex)

        combined = torch.cat([embeddings, view_emb, sex_emb], dim=1)
        return self.classifier(combined), outputs.attentions


def load_normal_model(checkpoint_path, device):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vision = AutoModel.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True
    ).vision_model

    model = CheXagentSigLIPBinary(vision)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
