import os
import numpy as np
import torch
import torch.nn as nn
import faiss
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "data/images"
DATA_DIR = "data"

EMBEDDINGS_PATH = "data/embeddings.npy"
IMAGE_PATHS_PATH = "data/image_paths.npy"
FAISS_INDEX_PATH = "data/jewelry.index"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier = nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_extensions = (".jpg", ".jpeg", ".png", ".webp")

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(root, file))

image_paths.sort()

if not image_paths:
    raise ValueError(f"No images found in {IMAGE_DIR}")

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
            print("Skipped:", image_path, e)

embeddings = np.asarray(embeddings, dtype="float32")
faiss.normalize_L2(embeddings)

relative_paths = [
    os.path.relpath(path, ".")
    for path in valid_image_paths
]

np.save(EMBEDDINGS_PATH, embeddings)
np.save(IMAGE_PATHS_PATH, np.asarray(relative_paths, dtype=object))

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

print("Done!")
print("Images:", len(relative_paths))
print("Embeddings:", embeddings.shape)
