import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import gradio as gr

# Load model from local directory
model_path = "segformer_leaf_model"

model = SegformerForSemanticSegmentation.from_pretrained(
    model_path,
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.to("cpu")
model.eval()

# Preprocessing
infer_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

def predict(image):
    H, W = image.shape[:2]

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed = infer_tf(image=img_bgr)
    tensor = processed["image"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        logits = outputs.logits

        logits_up = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        pred = torch.argmax(logits_up, dim=1).squeeze().numpy()

    colors = {
        0: (0, 0, 0),       # background
        1: (0, 255, 0),     # healthy
        2: (255, 0, 0)      # dry
    }

    mask = np.zeros((H, W, 3), dtype=np.uint8)

    for cls, color in colors.items():
        mask[pred == cls] = color

    overlay = cv2.addWeighted(image, 0.6, mask, 0.4, 0)

    return mask, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Mask"),
        gr.Image(type="numpy", label="Overlay (Mask + Image)")
    ],
    title="Leaf Health Segmentation (SegFormer)",
    description="Upload a leaf image to segment healthy vs dry areas."
)

demo.launch()