from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import AutoImageProcessor, AutoModel
import torch
import os
import numpy as np

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

dataset_folder = "dataset"

embeddings = []
image_names = []

for filename in os.listdir(dataset_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(dataset_folder, filename)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)

        embeddings.append(
            embedding.squeeze().numpy()
        )

        image_names.append(filename)

print("Total Images:", len(image_names))

embeddings = np.array(embeddings).astype("float32")

print("Embeddings Shape:", embeddings.shape)