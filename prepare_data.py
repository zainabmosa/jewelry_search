import os
import numpy as np
import torch
import torch.nn as nn
import faiss

from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)

# Find images
valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

image_paths = []

for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(root, file))

image_paths.sort()

print("Number of images:", len(image_paths))

if len(image_paths) == 0:
    raise ValueError("No images found. Put your images inside data/images.")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Remove the classification head
model.classifier = nn.Identity()

model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Extract embeddings
embeddings = []
valid_image_paths = []

with torch.no_grad():
    for image_path in tqdm(image_paths):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            embedding = model(tensor).cpu().numpy()[0]

            embeddings.append(embedding)
            valid_image_paths.append(image_path)

        except Exception as e:
            print("Skipped:", image_path, "|", e)

embeddings = np.asarray(embeddings, dtype="float32")

print("Embeddings shape:", embeddings.shape)

# Normalize vectors
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save relative paths
relative_image_paths = [
    os.path.relpath(path, PROJECT_DIR)
    for path in valid_image_paths
]

# Save files
np.save(
    os.path.join(DATA_DIR, "embeddings.npy"),
    embeddings
)

np.save(
    os.path.join(DATA_DIR, "image_paths.npy"),
    np.asarray(relative_image_paths, dtype=object)
)

faiss.write_index(
    index,
    os.path.join(DATA_DIR, "jewelry.index")
)

print("\nSaved successfully:")
print("data/embeddings.npy")
print("data/image_paths.npy")
print("data/jewelry.index")
