import torch
import gradio as gr
from transformers import AutoProcessor

from models.normal_model import load_normal_model
from models.tb_model import load_tb_model
from utils import register_pooling_hook, generate_heatmap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMAL_CKPT = r"checkpoint\version3.pt"
TB_CKPT     = r"checkpoint\best_model_attention_loss.pth"

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

# Processor
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# Load models (COMPLETELY SEPARATE)
normal_model = load_normal_model(NORMAL_CKPT, DEVICE)
tb_model     = load_tb_model(TB_CKPT, DEVICE)

normal_attn = register_pooling_hook(normal_model)
tb_attn     = register_pooling_hook(tb_model)

VIEW_MAP = {"AP": 1, "PA": 0, "Lateral": 2}
SEX_MAP  = {"Male": 0, "Female": 1}


# with both attention maps

# @torch.no_grad()
# def predict(image, view, sex):
#     image = image.convert("RGB")

#     inputs = processor(images=image, return_tensors="pt")
#     pixel_values = inputs["pixel_values"].to(DEVICE)

#     view_tensor = torch.tensor([VIEW_MAP[view]], device=DEVICE)
#     sex_tensor  = torch.tensor([SEX_MAP[sex]], device=DEVICE)

#     # -------- Normal / Abnormal --------
#     normal_logits, _ = normal_model(pixel_values, view_tensor, sex_tensor)
#     normal_prob = torch.sigmoid(normal_logits).item()
#     is_abnormal = normal_prob > 0.5

#     abnormal_heatmap_img = None
#     if is_abnormal:
#         abnormal_heatmap_img = generate_heatmap(
#             pixel_values,
#             normal_attn["weights"]
#         )

#     # -------- TB --------
#     tb_logits, _ = tb_model(pixel_values)
#     tb_prob = torch.sigmoid(tb_logits).item()
#     has_tb = tb_prob > 0.5

#     tb_heatmap_img = None
#     if has_tb:
#         tb_heatmap_img = generate_heatmap(
#             pixel_values,
#             tb_attn["weights"]
#         )

#     return (
#         "Abnormal" if is_abnormal else "Normal",
#         "TB Present" if has_tb else "No TB",
#         abnormal_heatmap_img,
#         tb_heatmap_img
#     )



# app = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Image(type="pil", label="Chest X-ray"),
#         gr.Dropdown(["AP", "PA", "Lateral"], label="View"),
#         gr.Dropdown(["Male", "Female"], label="Sex")
#     ],
#     outputs=[
#         gr.Textbox(label="Normal / Abnormal"),
#         gr.Textbox(label="TB Status"),
#         gr.Image(label="Abnormality Heatmap"),
#         gr.Image(label="TB Heatmap")
#     ],

#     title="CheXAgent – Chest X-ray Analysis",
# )

# if __name__ == "__main__":
#     app.launch(share=True)







# with TB attention map only

@torch.no_grad()
def predict(image, view, sex):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    view_tensor = torch.tensor([VIEW_MAP[view]], device=DEVICE)
    sex_tensor  = torch.tensor([SEX_MAP[sex]], device=DEVICE)

    # -------- Normal / Abnormal --------
    normal_logits, _ = normal_model(pixel_values, view_tensor, sex_tensor)
    normal_prob = torch.sigmoid(normal_logits).item()
    is_abnormal = normal_prob > 0.5

    # abnormal_heatmap_img = None
    # if is_abnormal:
    #     abnormal_heatmap_img = generate_heatmap(
    #         pixel_values,
    #         normal_attn["weights"]
    #     )

    # -------- TB --------
    tb_logits, _ = tb_model(pixel_values)
    tb_prob = torch.sigmoid(tb_logits).item()
    has_tb = tb_prob > 0.5

    tb_heatmap_img = None
    if has_tb:
        tb_heatmap_img = generate_heatmap(
            pixel_values,
            tb_attn["weights"]
        )

    return (
        "Abnormal" if is_abnormal else "Normal",
        "TB Present" if has_tb else "No TB",
        # abnormal_heatmap_img,
        tb_heatmap_img
    )



app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Chest X-ray"),
        gr.Dropdown(["AP", "PA", "Lateral"], label="View"),
        gr.Dropdown(["Male", "Female"], label="Sex")
    ],
    outputs=[
        gr.Textbox(label="Normal / Abnormal"),
        gr.Textbox(label="TB Status"),
        # gr.Image(label="Abnormality Heatmap"),
        gr.Image(label="TB Heatmap")
    ],

    title="CheXAgent-Chest X-ray Analysis",
)

if __name__ == "__main__":
    app.launch(share=True)