# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None

# from transformers import AutoImageProcessor, AutoModel
# import torch
# import os

# print("Loading DINOv2...")

# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# model = AutoModel.from_pretrained("facebook/dinov2-base")

# dataset_folder = "dataset"

# print("\nGenerating Embeddings...\n")

# for filename in os.listdir(dataset_folder):

#     if filename.lower().endswith((".jpg", ".jpeg", ".png")):

#         try:
#             image_path = os.path.join(dataset_folder, filename)

#             print(f"\nProcessing: {filename}")

#             image = Image.open(image_path).convert("RGB")

#             # Resize image
#             image = image.resize((224, 224))

#             inputs = processor(images=image, return_tensors="pt")

#             with torch.no_grad():
#                 outputs = model(**inputs)

#             embedding = outputs.last_hidden_state.mean(dim=1)

#             print("Shape:", embedding.shape)

#             # Print first 5 values of embedding
#             print("First 5 values:")
#             print(embedding[0][:5])

#             print("-" * 50)

#         except Exception as e:
#             print(f"Error processing {filename}")
#             print(e)

# print("\nAll images processed!")



import os

opt_files = set(os.listdir("dataset/optical"))
sar_files = set(os.listdir("dataset/sar"))

common = opt_files.intersection(sar_files)

print("Optical files:", len(opt_files))
print("SAR files:", len(sar_files))
print("Matching pairs:", len(common))

print("\nFirst 20 matches:")
for f in sorted(list(common))[:20]:
    print(f)